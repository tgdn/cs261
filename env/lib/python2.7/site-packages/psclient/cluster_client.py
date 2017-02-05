import atexit
from copy import copy
import functools
import os as _os
import glob as _glob
import requests as _requests
from requests import get, post
from time import sleep, mktime
from datetime import datetime, timedelta, tzinfo
from dateutil import parser
from ConfigParser import ConfigParser as _ConfigParser
from StringIO import StringIO as _StringIO
import tempfile as _tempfile
import shutil as _shutil
from subprocess import Popen, STDOUT
import boto.ec2.elb
import boto.ec2.cloudwatch
from boto.exception import BotoServerError
import boto.iam.connection
import json

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(_os.devnull, 'wb')

from boto import connect_s3
from boto.s3.key import Key

from psclient import file_util as _file_util
from psclient.file_util import parse_s3_path, s3_delete_key, exists, \
                            s3_recursive_delete, mkdir, copy_from_local
from .aws import _ec2_factory, _get_ec2_instances, _stop_instances, _ProductType

from .query_client import QueryClient
from .constants import _MAX_CREATE_TIMEOUT_SECS

import logging as _logging
_logger = _logging.getLogger(__name__)

ENV_TYPE = 'environment type'
# The maximum number of datapoints CloudWatch will return for a given query
CW_MAX_DATA_POINTS = 1440
# The minimum allowable period (seconds) for CloudWatch
CW_MIN_PERIOD = 60

# TODO: move all port to one location, and make sure pass the PORT to management server
DEFAULT_ADMIN_PORT = 8088
DEFAULT_ADMIN_UI_PORT = 8888
DEFAULT_QUERY_PORT = 9005

# Configuration section name and key name for product key
_PRODUCT_SECTION = 'Product'
_KEY_NAME = 'product_key'

# disable spurious output from requests library
_logging.getLogger("requests").setLevel(_logging.WARNING)

_ZERO = timedelta(0)
class _UTC(tzinfo):
    """
    A UTC datetime.tzinfo class modeled after the pytz library. It includes a
    __reduce__ method for pickling,
    """
    def fromutc(self, dt):
        if dt.tzinfo is None:
            return self.localize(dt)
        return super(_utc.__class__, self).fromutc(dt)

    def utcoffset(self, dt):
        return _ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return _ZERO

    def __reduce__(self):
        return _UTC, ()

    def __repr__(self):
        return "<UTC>"

    def __str__(self):
        return "UTC"

_utc = _UTC()

def _dt_to_utc_timestamp(t):
    if t.tzname() == 'UTC':
        return (t - datetime(1970, 1, 1, tzinfo=_utc)).total_seconds()
    elif not t.tzinfo:
        return mktime(t.timetuple())
    else:
        raise ValueError('Only local time and UTC time is supported')

class CertificateNotFoundError(Exception):
    pass

def retry(tries=3, delay=1, backoff=1, retry_exception=None):
    '''
    Retries a function or method until it has reached the maximum retries

    Parameters
    -----------
    tries : int, optional
        the number of times this function will retry

    delay : int, optional
        the number of seconds in delay to retry

    backoff : int, optional
        the number of factors by which the delay should increase after a retry

    retry_exception: Error, optional
        the type of error that only will trigger retries. Defaults to None so
        all types of error will trigger retries.

    This is derived from the original implementation of retry at:
    https://wiki.python.org/moin/PythonDecoratorLibrary#Retry
    '''
    def deco_retry(f):
        def f_retry(*args, **kargs):
            mtries, mdelay = tries, delay  # mutables
            while mtries > 1:
                try:
                    return f(*args, **kargs)  # run function
                except Exception as e:
                    if retry_exception and not isinstance(e, retry_exception):
                        break  # break and return f if exception caught is not expected
                    mtries -= 1  # decrease retry
                    sleep(mdelay)  # delay to next retry
                    mdelay *= backoff  # increase delay
            return f(*args, **kargs)  # last retry
        return f_retry
    return deco_retry

def predictive_service_environment_factory(env_info):
    env_type = env_info[ENV_TYPE]
    del env_info[ENV_TYPE]

    assert(env_type in ['TestStubEnv', 'LocalPredictiveServiceEnvironment', 'DockerPredictiveServiceEnvironment', 'Ec2PredictiveServiceEnvironment'])

    # Do appropriate type convertion
    if 'certificate_is_self_signed' in env_info and \
            env_info['certificate_is_self_signed'] is not None:
        value = env_info['certificate_is_self_signed'].lower()
        env_info['certificate_is_self_signed'] = (
            value == 'true' or value == 'yes' or value == '1')

    if(env_type == 'LocalPredictiveServiceEnvironment'):
        return LocalPredictiveServiceEnvironment(**env_info)
    elif(env_type == 'Ec2PredictiveServiceEnvironment'):
        return Ec2PredictiveServiceEnvironment(**env_info)
    elif(env_type == 'DockerPredictiveServiceEnvironment'):
        return DockerPredictiveServiceEnvironment(**env_info)

class PredictiveServiceEnvironment:

    def __init__(self, admin_port, admin_ui_port, port, aws_credentials):
        self.certificate_name = None
        self._request_schema = None
        self._should_verify_certificate = None
        self.port = port
        self.admin_port = admin_port
        self.admin_ui_port = admin_ui_port
        self.aws_credentials = aws_credentials

    def _get_all_hosts(self):
        return []

    def _poke_host(self, host_name, admin_key):
        _logger.info("Notifying: %s" % host_name)
        url = 'http://%s:%s/control/poke' % (host_name, self.port)
        self._post(url)

    def launch(self):
        pass

    def upload_file(self, local_path, remote_path, aws_credentials):
        '''
        Upload local file(s) to remote location
        '''
        is_dir = _os.path.isdir(local_path)
        _logger.info('Uploading %s "%s" to "%s"' % ('dir' if is_dir else 'file', local_path, remote_path))

        # Make sure the folder exists
        remote_folder = _os.path.split(remote_path)[0]
        if not exists(remote_folder):
            mkdir(remote_folder)

        copy_from_local(local_path, remote_path,
                        aws_credentials=aws_credentials,
                        is_dir = is_dir, silent = True)

    def remove_file(self, remote_path, aws_credentials):
        _logger.info('Removing file/folder:"%s"' % (remote_path))

        if _file_util.is_s3_path(remote_path):
            _file_util.s3_recursive_delete(remote_path, aws_credentials = aws_credentials)
        elif _file_util.is_hdfs_path(remote_path):
            _file_util.remove_hdfs(remote_path, recursive=True)
        else:
            files = _glob.glob(remote_path)
            for f in files:
                if _os.path.isdir(f):
                    _shutil.rmtree(f)
                else:
                    _os.remove(f)

    def list_file(self, remote_path, aws_credentials):
        if _file_util.is_local_path(remote_path):
            if _os.path.isdir(remote_path):
                return _os.listdir(remote_path)
            else:
                return []

        elif _file_util.is_s3_path(remote_path):
            files = _file_util.list_s3(remote_path, aws_credentials = aws_credentials)
        else:
            files = _file_util.list_hdfs(remote_path)

        if not files or len(files) == 0:
            return []

        return_value = set()

        # strip the relative path
        for f in files:
            rel_path = _os.path.relpath(f['path'], remote_path)

            # Only show first level children
            if rel_path != '.' and rel_path != '..':
                # In windows, the relpath could replace the separator with '\'
                return_value.add(rel_path.replace('\\', '/').split('/')[0])

        return_value = list(return_value)
        return_value.sort()
        return return_value

    def clear_connection(self):
        if hasattr(self, 'client_connection'):
            del self.client_connection

    def _get_predictive_service_client(self, api_key):
        if not hasattr(self, 'client_connection'):
            schema = self.__get_schema()
            endpoint = schema + self.load_balancer_dns_name
            verify_cert = hasattr(self, 'certificate_is_self_signed') and not self.certificate_is_self_signed
            self.client_connection = QueryClient(endpoint= endpoint, api_key=api_key,
                                                             should_verify_certificate = verify_cert)
        return self.client_connection

    def query(self, po_name, api_key, timeout, *args, **kwargs):
        self.client_connection = self._get_predictive_service_client(api_key)
        self.client_connection.set_query_timeout(timeout)
        return self.client_connection.query(po_name, *args, **kwargs)

    def feedback(self, request_id, api_key, timeout, **kwargs):
        self.client_connection = self._get_predictive_service_client(api_key)
        self.client_connection.set_query_timeout(timeout)
        return self.client_connection.feedback(request_id, kwargs)

    @staticmethod
    def _get_default_log_time_range(start_time, end_time):
        '''
        Get default time range for getting Predictive Service logs

        The default is past one hour
        '''
        if not start_time and not end_time:
            start_time = (datetime.now() - timedelta(hours=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0)
            end_time = datetime.utcnow()
        return (start_time, end_time)

    def terminate(self, delete_logs):
        pass

    def _post(self, url, data=None, admin_key=None):
        if data is None:
            data = {}

        if not self._should_verify_certificate:
            self._should_verify_certificate = self.__should_verify_certificate()

        headers = {'content-type': 'application/json'}
        response = post(url=url, data=json.dumps(data), headers=headers,
                verify=self._should_verify_certificate, timeout=60, auth=('admin_key', admin_key))
        if not response:
            raise RuntimeError("Request failed. Status code: %s" % response.status_code)
        return response

    def _ping(self):
        if not self._should_verify_certificate:
            self._should_verify_certificate = self.__should_verify_certificate()

        endpoint = self.__get_schema() + self.load_balancer_dns_name
        response = get(url=endpoint, params=None, verify=self._should_verify_certificate, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            raise RuntimeError("Error responding from service: response: %s" % str(response.text))

    def _get(self, url, params=None):
        if not self._should_verify_certificate:
            self._should_verify_certificate = self.__should_verify_certificate()

        return get(url=url, params=params, verify=self._should_verify_certificate, timeout=10)

    def __make_url(self, host_dns_name, endpoint):
        return "http://%s:%s/%s" % (host_dns_name,
                                    self.port,
                                    endpoint)

    def __cache_op_params(self, name, restart):
        data = {}
        if name:
            data.update({"name": name})

        if restart:
            data.update({"restart": restart})

        return data

    def _host_addrs(self):
        return ["localhost"]

    @staticmethod
    def _in_time_window(log_filename, start, end):
        """
        Check whether log file with name specified log_filename was created
        within the time window specified in the outer function.
        """
        #bucket.list might return the folder path.
        if log_filename[-4:] != ".log":
            return False

        _, log_filename = _os.path.split(log_filename)
        timestamp_str = log_filename.split(".")[0]

        # convert h-m-s to h:m:s
        date_time = timestamp_str.split("T")
        timestamp_str = date_time[0] + "T" + date_time[1].replace('-', ':')

        timestamp = parser.parse(timestamp_str)
        timestamp = timestamp.replace(tzinfo=_utc)
        timestamp = _dt_to_utc_timestamp(timestamp)

        if start and end:
            return timestamp >= start and timestamp <= end
        elif start:
            return timestamp >= start
        elif end:
            return timestamp <= end

        return True

    @staticmethod
    def _verify_state_path(state_path, aws_credentials = None):
        # Verify we're not overriding another predictive service.
        state_key = _os.path.join(state_path, 'state.ini')

        if _file_util.exists(state_key, aws_credentials):
            raise IOError("There is already a Predictive Service at the specified " \
                         "location. Use a different state path. If you want to load " \
                         "an existing Predictive Service, call 'load(...)'.")

    @staticmethod
    def _get_root_path(state_path):
        if _file_util.is_s3_path(state_path):
            bucket_name, key_name = parse_s3_path(state_path)
            if key_name:
                return "s3://%s/%s/" % (bucket_name, key_name)
            else:
                return "s3://%s/" % bucket_name
        else:
            if state_path[-1] != '/':
                return state_path + '/'
            else:
                return state_path

    def _write_state_config(self, state, state_path):
        state_key = _os.path.join(state_path, 'state.ini')

        if _file_util.is_s3_path(state_key):
            return self._write_s3_state_config(state, state_path)

        # for HDFS, create a local file and then upload
        is_remote = not _file_util.is_local_path(state_key)
        if is_remote:
            tmp_dir = _tempfile.mkdtemp(prefix='predictive_service_env')
            tmp_state_file = _os.path.join(tmp_dir, 'state.ini')
        else:
            tmp_state_file = state_key

        with open(tmp_state_file, 'w') as f:
            state.write(f)

        if _file_util.is_hdfs_path(state_key):
            _file_util.upload_to_hdfs(tmp_state_file, state_key, force=True)

        if is_remote:
            _shutil.rmtree(tmp_dir)

    def _write_s3_state_config(self, state, state_path):
        bucket_name, key_name = parse_s3_path(state_path)
        state_key = key_name + '/state.ini'

        # Write state file to S3
        state_fp = _StringIO()
        state.write(state_fp)
        state_fp.flush()
        state_fp.seek(0)
        conn = connect_s3(**self.aws_credentials)
        bucket = conn.get_bucket(bucket_name, validate=False)
        key = Key(bucket)
        key.key = state_key
        key.set_contents_from_file(state_fp)
        state_fp.close()

    @staticmethod
    def _get_s3_state_config(state_path, aws_credentials):
        bucket_name, key_name = parse_s3_path(state_path)
        state_key = key_name + '/state.ini'
        conn = connect_s3(**aws_credentials)
        bucket = conn.get_bucket(bucket_name, validate=False)
        key = bucket.get_key(state_key)

        if not key:
            raise IOError("No Predictive Service at the specified location.")

        contents = key.get_contents_as_string()
        config = _ConfigParser(allow_no_value=True)
        config.optionxform = str
        cont_fp = _StringIO(contents)
        cont_fp.seek(0)
        config.readfp(cont_fp)
        cont_fp.close()

        return config

    @staticmethod
    def _get_state_from_file(state_path, aws_credentials):
        state_key = _os.path.join(state_path, 'state.ini')

        if _file_util.is_s3_path(state_path):
            return PredictiveServiceEnvironment._get_s3_state_config(state_path, aws_credentials)

        # Download if it is remote
        is_remote = not _file_util.is_local_path(state_key)
        if is_remote:
            tmp_dir = _tempfile.mkdtemp(prefix='predictive_service_env')
            tmp_state_file = _os.path.join(tmp_dir, 'state.ini')
            if _file_util.is_hdfs_path(state_key):
                _file_util.download_from_hdfs(state_key, tmp_state_file)
            else:
                if not _os.path.isdir(state_path):
                    raise RuntimeError('%s must be a local folder.' % state_path)
        else:
            tmp_state_file = state_key

        config = _ConfigParser(allow_no_value=True)
        config.optionxform = str
        config.read(tmp_state_file)

        if is_remote:
            _shutil.rmtree(tmp_dir)

        return config

    @staticmethod
    def _get_product_key_from_s3(path, aws_credentials, is_state_path=False):
        bucket_name, key_name = parse_s3_path(path)
        state_path_key = key_name + '/license'
        conn = connect_s3(**aws_credentials)
        bucket = conn.get_bucket(bucket_name, validate=False)
        if is_state_path is True:
            key = bucket.get_key(state_path_key)
        else:
            key = bucket.get_key(key_name)

        if not key:
            raise IOError("No Predictive Service at the specified location.")

        try:
            contents = key.get_contents_as_string()
            config = _ConfigParser(allow_no_value=True)
            config.optionxform = str
            cont_fp = _StringIO(contents)
            cont_fp.seek(0)
            config.readfp(cont_fp)
            cont_fp.close()

            product_key = config.get(_PRODUCT_SECTION, _KEY_NAME)
            if product_key == -1:
                raise BaseException() # will fall into except block below
            else:
                # Don't clobber os.environ[PRODUCT_KEY_ENV] -- that would
                # prevent future calls to get_product_key from returning
                # different values. Each call should always check the
                # config file since it can change over time.
                return str(product_key).strip('"\'')
        except Exception as e:
            raise ValueError("Invalid Configuration file: %s" % e)

    @staticmethod
    def _get_product_key(config_path, aws_credentials, is_state_path=False):
        if _file_util.is_s3_path(config_path):
            return PredictiveServiceEnvironment._get_product_key_from_s3(
                                config_path, aws_credentials, is_state_path)

        if is_state_path is True:
            config_path = _os.path.join(config_path, 'license')

        # Download if it is remote
        is_remote = not _file_util.is_local_path(config_path)
        if is_remote:
            tmp_dir = _tempfile.mkdtemp(prefix='predictive_service_env')
            tmp_state_file = _os.path.join(tmp_dir, 'license')
            if _file_util.is_hdfs_path(config_path):
                _file_util.download_from_hdfs(config_path, tmp_state_file)
            else:
                if not _os.path.isdir(config_path):
                    raise RuntimeError('%s must be a local folder.' % config_path)
        else:
            tmp_state_file = config_path

        config = _ConfigParser(allow_no_value=True)
        config.optionxform = str
        config.read(tmp_state_file)

        if is_remote:
            _shutil.rmtree(tmp_dir)

        try:
            product_key = config.get(_PRODUCT_SECTION, _KEY_NAME)
            if product_key == -1:
                raise BaseException() # will fall into except block below
            else:
                # Don't clobber os.environ[PRODUCT_KEY_ENV] -- that would
                # prevent future calls to get_product_key from returning
                # different values. Each call should always check the
                # config file since it can change over time.
                return str(product_key).strip('"\'')
        except Exception as e:
            raise ValueError("Invalid Configuration file: %s" % e)

    def remove_state(self, state_path, deps_path, po_path):
        _logger.info('Deleting state data.')

        if _file_util.is_s3_path(state_path):
            self._remove_s3_state(state_path, deps_path, po_path)
        elif _file_util.is_hdfs_path(state_path):
            self._remove_hdfs_state(state_path, deps_path, po_path)
        else:
            self._remove_local_state(state_path, deps_path, po_path)

    def _remove_s3_state(self, state_path, deps_path, po_path):
        _logger.info('Deleting s3 state data.')
        bucket_name, key_name = parse_s3_path(state_path)
        state_key = key_name + '/state.ini'
        # make sure we have a valid s3 path
        if key_name:
            s3_root_path = "s3://%s/%s/" % (bucket_name, key_name)
        else:
            s3_root_path = "s3://%s/" % bucket_name
        # remove necessary folders and files
        try:
            s3_recursive_delete(s3_root_path + deps_path,
                                self.aws_credentials)
            s3_recursive_delete(s3_root_path + po_path,
                                self.aws_credentials)
            s3_delete_key(bucket_name, state_key, self.aws_credentials)
        except:
            _logger.error("Could not delete predictive object data from S3. "
                          "Please manually delete data under: %s" %
                          s3_root_path)

    def _remove_hdfs_state(self, state_path, deps_path, po_path):
        deps = _os.path.join(state_path, deps_path)
        po = _os.path.join(state_path, po_path)
        state_key = _os.path.join(state_path, 'state.ini')

        try:
            _file_util.remove_hdfs(deps)
            _file_util.remove_hdfs(po)
            _file_util.remove_hdfs(state_key)
        except Exception as e:
            _logger.error("Could not delete predictive object data from HDFS. %s" % e)

    def _remove_local_state(self, state_path, deps_path, po_path):
        if not _os.path.exists(state_path) or not _os.path.isdir(state_path):
            raise IOError("Cannot remote state at state path: %s" % state_path)

        deps = _os.path.join(state_path, deps_path)
        po = _os.path.join(state_path, po_path)
        state_key = _os.path.join(state_path, 'state.ini')
        # remove necessary folders and files
        try:
            _shutil.rmtree(deps)
            _shutil.rmtree(po)
            _os.remove(state_key)
        except:
            _logger.error("Could not delete predictive object data from state path. "
                          "Please manually delete data under: %s" %
                          state_path)

    def get_status(self, _show_errors=True):
        schema = self.__get_schema()
        endpoint = self.load_balancer_dns_name
        url = '%s%s/manage/status' % (schema, endpoint)
        try:
            response = self._post(url)
            info = json.loads(response.text)
            return info.values()
        except Exception as e:
            _logger.error('Could not get status for %s: %s' %
                          (url, e.message))
            raise

    @retry()
    def poke(self, admin_key):
        schema = self.__get_schema()
        endpoint = self.load_balancer_dns_name
        url = '%s%s/manage/poke' % (schema, endpoint)
        try:
            self._post(url, admin_key=admin_key)
        except Exception as e:
            _logger.error('Could not poke %s: %s' %
                          (endpoint, e.message))
            raise

    # Cache_enable is called right after instances are launched,
    # but the EC2 load balancer might not be ready to be accessed by
    # its dns name yet.
    # We give a higher delay than the default 1 second.
    # In Windows, it takes even longer for the DNS lookup to succeed, about 10m,
    # so we increase the number of retries to 30 times.
    @retry(tries=30, delay=30)
    def cache_enable(self, name, restart=True):
        schema = self.__get_schema()
        endpoint = self.load_balancer_dns_name
        url = '%s%s/manage/cache_enable' % (schema, endpoint)
        params = self.__cache_op_params(name, restart)
        try:
            self._post(url, data=params)
        except _requests.exceptions.ConnectionError:
            # This error 'nodename nor servname provided, or not known'
            # happens when ELB is not ready to be accessed yet.
            # We suppress printing this so that we do not confuse
            # the user.
            raise
        except Exception as e:
            _logger.error('Could not enable cache on %s: %s' %
                          (url, e.message))
            raise

    @retry()
    def cache_disable(self, name):
        schema = self.__get_schema()
        endpoint = self.load_balancer_dns_name
        url = '%s%s/manage/cache_disable' % (schema, endpoint)
        params = self.__cache_op_params(name, False)
        try:
            self._post(url, data=params)
        except Exception as e:
            _logger.error('Could not disable cache on %s: %s' %
                          (url, e.message))
            raise

    @retry()
    def cache_clear(self, name):
        schema = self.__get_schema()
        endpoint = self.load_balancer_dns_name
        url = '%s%s/manage/cache_clear' % (schema, endpoint)
        params = self.__cache_op_params(name, False)
        try:
            self._post(url, data=params)
        except Exception as e:
            _logger.error('Could not clear cache on %s: %s' %
                          (url, e.message))
            raise

    @retry()
    def flush_logs(self):
        schema = self.__get_schema()
        endpoint = self.load_balancer_dns_name
        url = '%s%s/manage/flush_logs' % (schema, endpoint)
        try:
            self._post(url)
        except Exception as e:
            _logger.error("Could not flush logs on %s: %s" %
                          (url, e.message))
            raise

    @retry()
    def reconfigure(self, system_conf):
        schema = self.__get_schema()
        endpoint = self.load_balancer_dns_name
        url = '%s%s/manage/reconfigure' % (schema, endpoint)
        post_body = system_conf.for_json()
        try:
            self._post(url, data=post_body)
        except Exception as e:
            _logger.error("Could not reconfigure host %s: %s"
                          % (url, e.message))
            raise

    def get_node_status(self, host_addr):
        url = self.__make_url(host_addr, "control/status")

        try:
            response = self._post(url)
        except Exception as e:  # TimeoutError, ConnectionError, etc.
            return {"error": "Cannot get status for host %s, error: %s"
                    % (host_addr, e.message)}

        try:
            _logger.debug(response.text)
            data = json.loads(response.text)
        except Exception as e:
            return {"error": "Cannot get status for host %s, error: %s"
                    % (host_addr, e.message)}

        data.update({"public_dns_name": host_addr})
        return data

    def __is_using_certificates(self):
        return hasattr(self, 'certificate_name') and self.certificate_name

    def __get_schema(self):
        if self._request_schema is None:
            self._request_schema = 'https://' if self.__is_using_certificates() else 'http://'
        return self._request_schema

    def __should_verify_certificate(self):
        return self.__is_using_certificates() and not (hasattr(self, 'certificate_is_self_signed') and self.certificate_is_self_signed)

    def _is_cache_ok(self, expected_cache_status, status=None):
        """
        Checks the state of the cache.

        Parameters
        ----------
        expected_cache_status : str ("healthy" or "disabled")
            The expected state of the system

        status : list[dict], optional
            The deployment's status as returned in a call to `get_status`. The
            list has as many elements as there are nodes in the deployment. Each
            element should contain the following fields: "cache", "dns_name",
            "id", "models", "reason", and "state". The "cache field is expected
            to be a dictionary. If left unspecfied (or set to None), we make a
            control-plane (/control/status) request to each node in the cluster.
        """
        status = status or self.get_status()

        # is cache either healthy or disabled on all nodes?
        def healthy_or_disabled(cache_status):
            if isinstance(cache_status, dict):
                if cache_status == {}:
                    cache_status = "disabled"
                elif cache_status.get("healthy"):
                    cache_status = "healthy"
                else:
                    cache_status = "unhealthy"

            if isinstance(cache_status, basestring):
                return cache_status

            raise RuntimeError("Unexpected value for cache_status: %s",
                               str(cache_status))

        node_cache_statuses = [healthy_or_disabled(x.get("cache", {})) \
                               for x in status]

        if len(node_cache_statuses) == 0:
            return True

        if len(set(node_cache_statuses)) > 1 \
          or "unhealthy" in node_cache_statuses:
            return False

        # is expected cache status the same as actual cache status?
        if expected_cache_status != node_cache_statuses[0]:
            return False

        # if cache is disabled globally, is it disabled for all models?
        for node_index, node_status in enumerate(status):
            node_cache_status = node_cache_statuses[node_index]
            model_caches = [model_status.get("cache_enabled") for model_status \
                            in node_status.get("models", [])]

            if node_cache_status == "disabled" and any(model_caches):
                return False

        # are all nodes using the same type of cache?
        cache_statuses = [x["cache"] for x in status]
        node_cache_types = {x.get("type", "nocache") if isinstance(x, dict) else "nocache" \
                            for x in cache_statuses}

        if len(node_cache_types) > 1:
            return False

        return True

class LocalPredictiveServiceEnvironment(PredictiveServiceEnvironment):
    def __init__(self, log_path, aws_credentials=None, num_hosts = 1, redis_manager = None,
                 web_server = None, graphlab_service = None,
                 admin_port = DEFAULT_ADMIN_PORT,
                 admin_ui_port = DEFAULT_ADMIN_UI_PORT,
                 port = DEFAULT_QUERY_PORT, **kwargs):
        PredictiveServiceEnvironment.__init__(self, admin_port, admin_ui_port, port, aws_credentials)
        self.load_balancer_dns_name = 'localhost:%s' % port
        self.log_path = log_path
        self.num_hosts = int(num_hosts)
        self.redis_manager = redis_manager
        self.web_server = web_server
        self.graphlab_service = graphlab_service
        self.certificate_is_self_signed = True

    def _host_addrs(self):
        return ["localhost"]

    def _host_infos(self):
        return {'localhost': {
            'id': 'localhost',
            'ip_address': '127.0.0.1',
            'private_ip_address': '127.0.0.1',
            'state': 'InService',
            'reason': 'N/A',
        }}

    def _get_state(self):
        result = {}
        result[ENV_TYPE] = 'LocalPredictiveServiceEnvironment'
        result['num_hosts'] = self.num_hosts
        result['log_path'] = self.log_path
        result['port'] = self.port
        result['admin_port'] = self.admin_port
        result['admin_ui_port'] = self.admin_ui_port

        return result

    def _is_cache_ok(self, expected_cache_status, status=None):
        # we should implement this
        return True

    def terminate(self, delete_logs):
        _logger.info('Terminating service.')
        try:
            if self.web_server:
                _logger.info('Terminating web server.')
                self.web_server.terminate()
        except:
            pass

        try:
            if self.graphlab_service:
                _logger.info('Terminating graphlab_service.')
                self.graphlab_service.terminate()
        except:
            pass

        try:
            if self.redis_manager:
                _logger.info('Terminating redis manager.')
                self.redis_manager.terminate()
        except:
            pass

        if delete_logs:
            _logger.info('Deleting log files.')
            try:
                s3_recursive_delete(self.log_path)
            except:
                _logger.info("Could not delete log file. Please manually delete files under: %s"
                             % self.log_path)

    @staticmethod
    def launch(predictive_service_path, log_path, num_hosts, port):
        node_manager_dir = _os.environ['NODE_MANAGER_ROOT']

        # Start Redis for caching
        env_vars = copy(_os.environ)
        env_vars['PYTHONPATH'] = _os.pathsep.join([env_vars['PYTHONPATH'], '.'])
        env_vars['PREDICTIVE_SERVICE_STATE_PATH'] = predictive_service_path
        env_vars['LOG_PATH'] = log_path

        env_vars['REDIS_TRIB_PATH'] = _os.environ['REDIS_TRIB_PATH']
        env_vars['REDIS_SERVER_PATH'] = _os.environ['REDIS_SERVER_PATH']

        redis_manager = Popen(
            ['python', 'redismanager/redis_manager.py', str(max(3, num_hosts))],
            cwd = node_manager_dir, env = env_vars, stdout=DEVNULL, stderr=STDOUT)

        _logger.info(
            "Running Redis manager with PID: %d" % (redis_manager.pid))

        # Start the Predictive Service
        web_server = Popen(
            ['python', 'psws/ps_server.py'],
            cwd = node_manager_dir, env = env_vars, stdout=DEVNULL, stderr=STDOUT)

        _logger.info(
            "Running web server with PID: %d" % (web_server.pid))

        # Start GraphLab Service
        graphlab_service = Popen(
            ['python', 'glservice/graphlab_service.py'],
            cwd = node_manager_dir, env = env_vars,  stdout=DEVNULL, stderr=STDOUT)

        _logger.info(
            "Running GraphLab Create service with PID: %d" % (graphlab_service.pid))

        instance = LocalPredictiveServiceEnvironment(
            log_path, port, num_hosts, redis_manager = redis_manager,
            web_server = web_server, graphlab_service = graphlab_service)

        atexit.register(functools.partial(instance.terminate, False))

        return instance

class DockerPredictiveServiceEnvironment(PredictiveServiceEnvironment):

    def __init__(self,
        load_balancer_dns_name,
        log_path,
        certificate_name = None,
        certificate_is_self_signed = True,
        aws_credentials = None,
        port = DEFAULT_QUERY_PORT,
        admin_port = DEFAULT_ADMIN_PORT,
        admin_ui_port = DEFAULT_ADMIN_UI_PORT,
        load_balancer_stats_port = DEFAULT_QUERY_PORT - 1,
        metrics_server = None):
        PredictiveServiceEnvironment.__init__(self, admin_port, admin_ui_port, port, aws_credentials)
        self.load_balancer_dns_name = load_balancer_dns_name
        self.log_path = log_path
        self.ctl_port = load_balancer_stats_port
        self.metrics_server = metrics_server
        self.certificate_name = certificate_name
        self.certificate_is_self_signed = certificate_is_self_signed

    def __repr__(self):
        ret = ""
        ret += 'DockerPredictiveServiceEnvironment:\n'
        ret += '\tload_balancer_dns_name: %s\n' % str(self.load_balancer_dns_name)
        ret += '\tlog_path: %s\n' % str(self.log_path)
        return ret

    def __str__(self):
        return self.__repr__()

    def _get_state(self):
        result = {}
        result[ENV_TYPE] = 'DockerPredictiveServiceEnvironment'
        result['log_path'] = self.log_path
        result['load_balancer_dns_name'] = self.load_balancer_dns_name
        result['load_balancer_stats_port'] = self.ctl_port
        result['metrics_server'] = self.metrics_server
        result['certificate_is_self_signed'] = self.certificate_is_self_signed
        result['certificate_name'] = self.certificate_name
        result['port'] = self.port
        result['admin_port'] = self.admin_port
        result['admin_ui_port'] = self.admin_ui_port
        return result

    def _host_infos(self):
        result = {}
        for host in self._get_all_hosts():
            ip_address, port = host['svname'].split(':')
            smalldict = {}
            smalldict['id'] = ip_address
            smalldict['ip_address'] = ip_address
            smalldict['private_ip_address'] = ip_address
            smalldict['state'] = host['status']
            smalldict['reason'] = host['last_chk']
            result[ip_address] = smalldict
        return result

    def _check_metrics_name(self, name):
        '''
        Checks if the metrics name is valid or not
        '''
        name_list = ['requests', 'latency', 'cache::hits', 'cache::misses', 'cache::latency',
                     'cache::num_keys', 'num_hosts_in_cluster', 'num_objects_queryable',
                     'num_objects_registered', 'gls::latency', 'exceptions',
                     'diskspace_root', 'diskspace_tmp', 'memory_available',
                     'memory_total', 'connection_count', 'connection_max']

        return name in name_list

    def _format_metrics(self, metric_name, sf):
        '''
        Format the raw metrics SFrame.
        '''
        # get rename columns (value -> sum/maximum) and list of columns to be returned
        rename_columns = {}
        columns = ['time', 'unit']
        if metric_name in ['num_hosts_in_cluster', 'num_objects_queryable',
                           'num_objects_registered']:
            rename_columns['value'] = 'maximum'
            columns.append('maximum')
        elif 'latency' not in metric_name:
            rename_columns['value'] = 'sum'
            columns.append('sum')
        else:
            columns.append('average')

        # rename columns
        sf.rename(rename_columns)

        # get only the needed columns
        sf = sf[columns]

        # convert time column to datetime column
        sf['time'] = sf['time'].apply(lambda x : datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

        return sf

    def terminate(self, remove_logs):
        _logger.warn("Cannot terminated your Docker Predictive Service from client.")

    def terminate_instances(self, instance_ids):
        _logger.warn("Cannot terminate instance(s) on Docker Predictive Service from client.")

    def add_instances(self, instance_ids):
        _logger.warn("Cannot add instance(s) on Docker Predictive Service from client.")

    def _get_all_hosts(self):
        hostname, port = self.load_balancer_dns_name.split(':')
        ctl_dns = 'http://' + hostname + ':' + str(self.ctl_port)
        stats_url = ctl_dns + '/stats;csv'
        stats_resp = self._get(stats_url)

        # clean up raw status
        stats = stats_resp.content[2:]  # remove the '# ' in the start
        stats = stats.split('\n')
        column_names = stats[0].split(',')
        stats_list = []
        for row in stats[1:]:
            r = row.split(',')
            if len(r) != len(column_names):
                continue
            stat = {column_names[idx]: r[idx] for idx in range(0, len(column_names))}
            stats_list.append(stat)

        # obtain ps server status
        ps_server_status = []
        for st in stats_list:
            if st['pxname'] == 'ps_servers' and st['svname'] != 'BACKEND':
                ps_server_status.append(st)

        return ps_server_status

    def _host_addrs(self):
        result = []
        server_status = self._get_all_hosts()
        for stat in server_status:
            hostname, port = stat['svname'].split(':')
            result.append(hostname)

        return result

class Ec2PredictiveServiceEnvironment(PredictiveServiceEnvironment):

    def __init__(self, load_balancer_dns_name, region, log_path,
                 certificate_name, certificate_is_self_signed, aws_credentials,
                 admin_port = DEFAULT_ADMIN_PORT,
                 admin_ui_port = DEFAULT_ADMIN_UI_PORT,
                 port = DEFAULT_QUERY_PORT):
        PredictiveServiceEnvironment.__init__(self,
            admin_port = admin_port, admin_ui_port=admin_ui_port, port=port, aws_credentials=aws_credentials)
        self.region = region
        self.load_balancer_dns_name = load_balancer_dns_name
        self.log_path = log_path
        self.certificate_name = certificate_name
        self.certificate_is_self_signed = certificate_is_self_signed

    def __repr__(self):
        ret = ""
        ret += 'Ec2PredictiveServiceEnvironment:\n'
        ret += '\tload_balancer_dns_name: %s\n' % str(self.load_balancer_dns_name)
        ret += '\tregion: %s\n' % str(self.region)
        ret += '\tlog_path: %s\n' % str(self.log_path)
        ret += '\tcertificate_name: %s\n' % str(self.certificate_name)
        if self.certificate_name:
            ret += '\tcertificate_is_self_signed: %s\n' % str(self.certificate_is_self_signed)
        return ret

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def launch(name, ec2_config, s3_predictive_object_path, num_hosts,
               ssl_credentials, aws_credentials, started, admin_port,
               admin_ui_port, product_key, scheme):

        def launch_hosts(num_hosts):
            return _ec2_factory(ec2_config.instance_type,
                product_key,
                region = ec2_config.region,
                CIDR_rule = cidr_ip,
                security_group_name = ec2_config.security_group,
                tags = tags,
                user_data = user_data,
                credentials = aws_credentials,
                num_hosts = num_hosts,
                ami_service_parameters = {'service': 'predictive'},
                additional_ports_to_open = [admin_port, admin_ui_port],
                product_type = _ProductType.PredictiveServices,
                subnet_id = ec2_config.subnet_id,
                security_group_id = ec2_config.security_group_id)

        # Verify we're not overriding another predictive service.
        PredictiveServiceEnvironment._verify_state_path(s3_predictive_object_path,
                                                           aws_credentials)

        s3_log_path = "%s/logs" % (s3_predictive_object_path)
        user_data = {
            'aws_access_key': aws_credentials['aws_access_key_id'],
            'aws_secret_key': aws_credentials['aws_secret_access_key'],
            'predictive_service_state_path': s3_predictive_object_path,
            'num_hosts': num_hosts,
            'consul_action': 'bootstrap',
            'consul_master_ip': ''
        }

        # add tags for all EC2 instances to indicate they are related to this Predictive Service
        tags = {}
        if ec2_config.tags:
            tags.update(ec2_config.tags)
        tags.update({'Name': name, 'predictive_service': name})

        cidr_ip = ec2_config.cidr_ip if hasattr(ec2_config, 'cidr_ip') else None

        _logger.info("[Step 1/5]: Launching EC2 instances.")

        # Start the hosts up.
        first_host, security_group, subnet_id = launch_hosts(num_hosts = 1)
        user_data.update({'consul_master_ip': first_host.instance.private_ip_address})

        # Start the rest hosts up.
        if num_hosts > 1:
            ec2_hosts, security_group, subnet_id = launch_hosts(num_hosts = num_hosts - 1)
            if num_hosts - 1 == 1:
                ec2_hosts = [first_host, ec2_hosts]
            else:
                ec2_hosts.append(first_host)
        else:
            ec2_hosts = [first_host]

        lb = None
        try:
            # Determine host ids and availability zones used.
            zones, host_ids, is_vpc = set(), [], False
            for i in ec2_hosts:
                zones.add(i.instance.placement)
                host_ids.append(i.instance_id)

            is_vpc = bool(ec2_hosts[0].instance.vpc_id)

            certificate_name, certificate_is_self_signed = None, None
            if ssl_credentials:
                # Using HTTPS
                (private_key_path, public_certificate_path, certificate_is_self_signed) = ssl_credentials
                certificate_name = name
                certificate_id = Ec2PredictiveServiceEnvironment._upload_ssl_info(certificate_name,
                                                                                  private_key_path,
                                                                                  public_certificate_path,
                                                                                  aws_credentials)
                listener_tuple = [(443, DEFAULT_QUERY_PORT, 'https', certificate_id),
                                  (admin_port, DEFAULT_ADMIN_PORT, 'https', certificate_id),
                                  (admin_ui_port, DEFAULT_ADMIN_UI_PORT, 'https', certificate_id)]
            else:
                # Using HTTP
                _logger.info("WARNING: Launching Predictive Service without SSL certificate!")
                listener_tuple = [(80, DEFAULT_QUERY_PORT, 'http'),
                                  (admin_port, DEFAULT_ADMIN_PORT, 'http'),
                                  (admin_ui_port, DEFAULT_ADMIN_UI_PORT, 'http')]

            conn = boto.ec2.elb.connect_to_region(ec2_config.region, **aws_credentials)

            # Create the load balancer.
            _logger.info("[Step 2/5]: Launching Load Balancer.")

            lb = Ec2PredictiveServiceEnvironment._create_load_balancer(conn, name, subnet_id, zones, scheme, listener_tuple)

            if is_vpc:
                lb.apply_security_groups([security_group.id])

            _logger.info("[Step 3/5]: Configuring Load Balancer.")

            # Configure healthCheck
            health_target = "http:%s/control/healthcheck" % DEFAULT_QUERY_PORT
            health_check = boto.ec2.elb.HealthCheck(interval=20, healthy_threshold=3,
                                                    unhealthy_threshold=5,
                                                    target=health_target)
            lb.configure_health_check(health_check)

            # Add EC2 instances to the load balancer.
            lb.register_instances(host_ids)

        except Exception as e:
            _logger.error("Could not create or configure the load balancer, terminating EC2 instances." \
                "Exception: %s" % e.message)
            #TODO: _stop_instances can raise exception
            _stop_instances([h.instance_id for h in ec2_hosts], ec2_config.region,
                            credentials = aws_credentials)

            if lb:
                _logger.info("Deleting the load balancer.")
                lb.delete()

            raise

        return Ec2PredictiveServiceEnvironment(lb.dns_name, ec2_config.region, s3_log_path,
                                               certificate_name, certificate_is_self_signed,
                                               aws_credentials, admin_port=admin_port,
                                               admin_ui_port=admin_ui_port)

    @staticmethod
    @retry(tries=10, delay=1, backoff=2, retry_exception=CertificateNotFoundError)
    def _create_load_balancer(conn, name, subnet_id, zones, scheme, listener_tuple):
        '''
        Create a load balancer with retries
        '''
        try:
            if subnet_id is None:
                lb = conn.create_load_balancer(name, zones, listener_tuple)
            else:
                lb = conn.create_load_balancer(name, zones = None, listeners = listener_tuple, scheme = scheme, subnets = [subnet_id])
            return lb
        except BotoServerError as e:
            # We just uploaded the certificate, so there's a good chance it will not be found, yet.
            if "<Code>CertificateNotFound</Code>" in str(e):
                raise CertificateNotFoundError()
            else:
                raise

    def _get_metric_info(self, name):
        info = dict()
        info['unit'] = 'Seconds' if 'latency' in name else 'Count'
        if 'latency' in name:
            info['statistics'] = 'Average'
        elif name in ['num_hosts_in_cluster', 'num_objects_queryable', 'num_objects_registered']:
            info['statistics'] = 'Maximum'
        else:
            info['statistics'] = 'Sum'
        return info

    def _check_metrics_name(self, name):
        '''
        Checks if the metrics name is valid or not
        '''
        name_list = ['requests', 'latency', 'cache::hits', 'cache::misses', 'cache::latency',
                     'cache::num_keys', 'num_hosts_in_cluster', 'num_objects_queryable',
                     'num_objects_registered', 'gls::latency', 'HealthyHostCount', 'exceptions',
                     'diskspace_root', 'diskspace_tmp', 'memory_available',
                     'memory_total', 'connection_count', 'connection_max']
        return name in name_list

    def _rename_columns(self, name_list):
        '''
        Rename columns:
            - 'Timestamp' -> 'time'
            - convert all column names to lowercase str
        '''
        new_columns = {}
        for i in name_list:
            if i == 'Timestamp':
                new_columns[i] = 'time'
            else:
                new_columns[i] = i.lower()
        return new_columns

    def _get_cloudwatch_handle(self):
        try:
            conn = boto.ec2.cloudwatch.connect_to_region(self.region, **self.aws_credentials)
            return conn
        except Exception:
            _logger.error("Unable to connect to to CloudWatch in region '%s'" % self.region)
        raise Exception("Cannot connect to CloudWatch in region '%s'." % self.region)

    def _get_load_balancer_handle(self):
        try:
            conn = boto.ec2.elb.connect_to_region(self.region, **self.aws_credentials)
            for i in conn.get_all_load_balancers():
                if i.dns_name == self.load_balancer_dns_name:
                    return i
        except Exception as e:
            _logger.error("Unable to connect to ELB with name '%s' in region '%s', error: %s"
                          % (self.load_balancer_dns_name, self.region, e))

        raise Exception("Cannot find load balancer with name '%s' in region '%s'."
                        % (self.load_balancer_dns_name, self.region))

    def _host_infos(self):
        result = {}
        load_balancer = self._get_load_balancer_handle()

        for host in load_balancer.get_instance_health():
            host_id = host.instance_id
            host_instance = _get_ec2_instances([host_id], self.region,
                                               aws_credentials = self.aws_credentials)
            dns_name = host_instance[0].public_dns_name

            smalldict = {}
            smalldict['id'] = host_id
            smalldict['ip_address'] = host_instance[0].ip_address
            private_ip_address = host_instance[0].private_ip_address
            smalldict['private_ip_address'] = private_ip_address
            smalldict['state'] = host.state
            smalldict['reason'] = host.reason_code
            result[private_ip_address] = smalldict
        return result

    def _get_state(self):
        result = {}
        result[ENV_TYPE] = 'Ec2PredictiveServiceEnvironment'
        result['load_balancer_dns_name'] = self.load_balancer_dns_name
        result['region'] = self.region
        result['certificate_name'] = self.certificate_name
        result['log_path'] = self.log_path
        result['certificate_is_self_signed'] = self.certificate_is_self_signed
        result['port'] = self.port
        result['admin_port'] = self.admin_port
        result['admin_ui_port'] = self.admin_ui_port

        return result

    @retry()
    def _get_instance_attributes(self, instance_id = None):
        conn = boto.ec2.connect_to_region(self.region, **self.aws_credentials)

        if not instance_id:
            hosts = self._get_all_hosts()
            if len(hosts) == 0:
                raise RuntimeError('There is no hosts available!')

            # pick first one
            instance_id = hosts[0].id

        reservations = conn.get_all_instances([instance_id])
        instance = reservations[0].instances[0]

        # get instance type
        instance_type = instance.instance_type

        # get instance security group
        security_groups = instance.groups
        security_group = None
        if len(security_groups) > 0:
            security_group = security_groups[0].name

        # get instance tags
        tags = instance.tags or {}

        return {"instance_type": instance_type,
                "security_group_name": security_group,
                "tags": tags,
                "security_group_id": security_groups[0].id,
                "subnet_id": instance.subnet_id,
                "private_ip_address": instance.private_ip_address}

    def terminate_instances(self, instance_ids):
        # terminate specified instances
        try:
            _stop_instances(instance_ids, self.region,
                            credentials=self.aws_credentials)
        except Exception:
            _logger.error("Could not stop instances %s. Please terminate " \
                          "the instances manually if they're still running."
                          % str(instance_ids))

        # remove specified instances from the load balancer
        try:
            load_balancer = self._get_load_balancer_handle()
            load_balancer.deregister_instances(instance_ids)
        except:
            _logger.error("Could not remove instances %s from load balancer."
                          % str(instance_ids))

    def _get_load_balancer_status(self):
        result = []
        load_balancer = self._get_load_balancer_handle()
        for host in load_balancer.get_instance_health():
            host_id = host.instance_id
            host_instance = _get_ec2_instances([host_id], self.region,
                                               aws_credentials = self.aws_credentials)

            result.append({
                'id': host_id,
                'ip_address': host_instance[0].ip_address,
                'private_ip_address': host_instance[0].private_ip_address,
                'state': host.state,
                'reason': host.reason_code,
            })
        return result

    @retry()
    def add_instances(self, s3_state_path, num_hosts, instance_type,
                      tags,
                      security_group_id,
                      subnet_id,
                      CIDR_rule=None,
                      additional_ports_to_open=None,
                      existing_node_private_ip=None):
        # get a load balancer handle, which we'll use for:
        #   1. getting an availability zone if one wasn't passed in
        #   2. getting the current cluster instances
        #   3. adding the replacement node to the load balancer
        lb = self._get_load_balancer_handle()
        availability_zone = lb.availability_zones[0]

        user_data = {
            'aws_access_key': self.aws_credentials['aws_access_key_id'],
            'aws_secret_key': self.aws_credentials['aws_secret_access_key'],
            'predictive_service_state_path': s3_state_path,
            'consul_action': 'server',
            'consul_master_ip': existing_node_private_ip}

        # get product key
        product_key = PredictiveServiceEnvironment._get_product_key(
                                        s3_state_path, self.aws_credentials,
                                        is_state_path=True)

        # launch new instance
        instance_ids = []
        try:
            ec2_hosts, security_group, subnet_id = _ec2_factory(instance_type,
                                    product_key,
                                    region=self.region,
                                    availability_zone=availability_zone,
                                    CIDR_rule=CIDR_rule,
                                    tags=tags, user_data=user_data,
                                    credentials=self.aws_credentials,
                                    num_hosts=num_hosts,
                                    ami_service_parameters={'service': 'predictive'},
                                    additional_ports_to_open=additional_ports_to_open,
                                    product_type=_ProductType.PredictiveServices,
                                    subnet_id = subnet_id,
                                    security_group_id = security_group_id)
            # add to the load balancer
            if not isinstance(ec2_hosts, list):
                ec2_hosts = [ec2_hosts]

            instance_ids = [i.instance_id for i in ec2_hosts]
            lb.register_instances(instance_ids)

            # poll load balancer until instances are available
            starttime = datetime.now()

            def seconds_elapsed():
                return (datetime.now() - starttime).total_seconds()

            while seconds_elapsed() < _MAX_CREATE_TIMEOUT_SECS:
                # query status, verify all InService
                nodes = self._get_load_balancer_status()
                nodes_in_service = [node["state"] == "InService" for node in nodes]

                if all(nodes_in_service):
                    _logger.info("Cluster is fully operational, [%d/%d] " \
                                 "instances currently in service.",
                                 nodes_in_service.count(True),
                                 len(nodes_in_service))
                    break
                else:
                    _logger.info("Cluster not fully operational yet, [%d/%d] " \
                                 "instances currently in service.",
                                 nodes_in_service.count(True),
                                 len(nodes_in_service))
                    sleep(15)

            if not all(nodes_in_service):
                raise RuntimeError("Unable to successfully add new instances to " \
                                   "load balancer. Please confirm in AWS " \
                                   "Management Console")
        except Exception as e:
            _logger.error(e.message)
            if instance_ids:
                self.terminate_instances(instance_ids)
            raise

    def terminate(self, delete_logs):
        try:
            load_balancer_handle = self._get_load_balancer_handle()
            conn = boto.ec2.elb.connect_to_region(self.region, **self.aws_credentials)
            _logger.info("Deleting load balancer: %s" % load_balancer_handle.name)
            conn.delete_load_balancer(load_balancer_handle.name)
        except:
            _logger.error("Could not delete load balancer. Please manually delete the following load "
                      "balancer: %s" % self.load_balancer_dns_name)

        # Terminate all hosts
        try:
            host_ids = [i.id for i in load_balancer_handle.instances]
            _stop_instances(host_ids, self.region, credentials = self.aws_credentials)
        except:
            _logger.error("Could not terminate hosts. Please manually terminate from the AWS console.")

        if delete_logs:
            _logger.info('Deleting log files.')
            try:
                s3_recursive_delete(self.log_path, self.aws_credentials)
            except:
                _logger.info("Could not delete log file. Please manually delete files under: %s"
                             % self.log_path)

        # Delete the server certificate
        if self.certificate_name:
            self._delete_certificate()

    @retry(tries=10, delay=1, backoff=2)
    def _delete_certificate(self):
        conn = boto.iam.connection.IAMConnection(**self.aws_credentials)
        conn.delete_server_cert(self.certificate_name)

    def _get_all_hosts(self):
        load_balancer_handle = self._get_load_balancer_handle()

        host_ids = [i.id for i in load_balancer_handle.instances]

        if len(host_ids) > 0:
            return _get_ec2_instances(host_ids, self.region, self.aws_credentials)

        return []

    def _host_addrs(self):
        addrs = []
        for inst in self._get_all_hosts():
            addrs.append(inst.private_ip_address)
        return addrs

    @staticmethod
    def _upload_ssl_info(certificate_name, private_key_path, public_certificate_path,
                         credentials):
        # Read in private key and public certificate
        with open(_file_util.expand_full_path(private_key_path), 'r') as file:
            private_key = file.read()
        with open(_file_util.expand_full_path(public_certificate_path), 'r') as file:
            cert = file.read()

        # Upload private key and public certificate
        conn = boto.iam.connection.IAMConnection(**credentials)

        try:
            resp = conn.upload_server_cert(certificate_name, cert, private_key)
        except BotoServerError as e:
            if not hasattr(e, 'status') or e.status != 409:
                raise

            _logger.warn("Error trying to upload server certificate: %s" % e)

            # Handle the case where the certificate already exists
            # delete the conflict certificate and try again
            _logger.info('Deleting existing certificate %s.' % certificate_name)
            conn.delete_server_cert(certificate_name)
            resp = conn.upload_server_cert(certificate_name, cert, private_key)

        # Get the certificate id
        certificate_id = resp['upload_server_certificate_response'] \
            ['upload_server_certificate_result']['server_certificate_metadata']['arn']

        return certificate_id

class TestEc2Env(Ec2PredictiveServiceEnvironment):

    class Host(object):
        def __init__(self):
            self.private_ip_address = 0

    def __init__(self, aws_credentials):
        Ec2PredictiveServiceEnvironment.__init__(self,
                'load_balancer_dns_name', 'us-west-2', 's3://log_path',
                None, None, aws_credentials)

    def poke(self, admin_key):
        pass

    def cache_enable(self, name, restart):
        pass

    def cache_disable(self, name):
        pass

    def cache_clear(self, name):
        pass

    def reconfigure(self, system_conf):
        pass

    def terminate(self, delete_logs):
        pass

    def _get_all_hosts(self):
        return [TestEc2Env.Host()]

    def launch(self, state_path):
        # Verify we're not overriding another predictive service.
        PredictiveServiceEnvironment._verify_state_path(state_path, self.aws_credentials)

    def get_status(self):
        pass

    def _host_addrs(self):
        return ["localhost"]

class TestDockerEnv(DockerPredictiveServiceEnvironment):

    def __init__(self, load_balancer_dns_name, log_path):
        DockerPredictiveServiceEnvironment.__init__(self, load_balancer_dns_name,
            log_path)

    def poke(self, admin_key):
        pass

    def cache_enable(self, name, restart):
        pass

    def cache_disable(self, name):
        pass

    def cache_clear(self, name):
        pass

    def reconfigure(self, system_conf):
        pass

    def terminate(self, delete_logs):
        pass

    def _get_all_hosts(self):
        return range(0, self.num_hosts)

    def launch(self, state_path):
        # Verify we're not overriding another predictive service.
        PredictiveServiceEnvironment._verify_state_path(state_path)

    def get_status(self):
        pass

    def _host_addrs(self):
        return ["localhost"]
