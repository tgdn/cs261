'''
An instance of Predictive Service can be created using create()
'''
import time as _time
from re import compile as _compile
from uuid import uuid4 as _random_uuid
import os as _os
import datetime as _datetime

from boto.ec2.elb import connect_to_region as _lb_connection
from boto.exception import S3ResponseError as _S3ResponseError

# since _predictive_service_environment imports these, need to have them defined first
from .constants import (
    _MAX_CREATE_TIMEOUT_SECS,
    _DEFAULT_ADMIN_PORT,
    _DEFAULT_ADMIN_UI_PORT,
    _NODE_LAUNCH_LIMIT,
    _CACHE_TTL_ON_UPDATE_SECS,
    _CACHE_MAX_MEMORY_MB,
    _CACHE_QUERY_TTL_SECS,
    _CACHE_FEEDBACK_TTL_SECS,
    _ADHOC_EXECUTE,
)

_ALL_SUPPORTED_CONFIGS = [
    _CACHE_TTL_ON_UPDATE_SECS,
    _CACHE_QUERY_TTL_SECS,
    _CACHE_FEEDBACK_TTL_SECS,
    _CACHE_MAX_MEMORY_MB,
    _ADHOC_EXECUTE,
]

from .cluster_client import Ec2PredictiveServiceEnvironment as _Ec2PredictiveServiceEnvironment

import psclient
import psclient.file_util as _file_util
from .state_util import generate_init_state_file as _generate_init_state_file

from .predictive_service_client import PredictiveService as _PredictiveService

from ._ec2_config import Ec2Config

import logging as _logging
_logger = _logging.getLogger(__name__)

from .config import generate_config

_name_checker = _compile('^[a-zA-Z-]+$')

from collections import namedtuple

def _save_initial_state(environment, state, state_path):
    if not environment:
        raise ValueError("Cannot save initial state: invalid environment.")
    environment._write_state_config(state, state_path)

def _check_name_and_port(name, port):
    '''
    Check for invalid name and ports
    '''
    if type(name) != str:
        raise TypeError("Name of Predictive Service needs to be a string")

    if (port < 0 or port > 65535):
        raise ValueError("Port must be within a certain range, 0 to 65535.")
    if (port == 9006 or port == 19006):
        raise ValueError("Port 9006 and 19006 are reserved for cache. Please use another port.")

class SystemConfig(namedtuple("SystemConfig",
                              [_CACHE_TTL_ON_UPDATE_SECS,
                               _CACHE_MAX_MEMORY_MB,
                               _CACHE_QUERY_TTL_SECS,
                               _CACHE_FEEDBACK_TTL_SECS,
                               _ADHOC_EXECUTE])):
    def __new__(cls, cache_ttl_on_update_secs=None, cache_max_memory_mb=None,
                cache_query_ttl_secs=None, cache_feedback_ttl_secs=None,
                adhoc_execute=None):
        if cache_ttl_on_update_secs is not None:
            cache_ttl_on_update_secs = int(cache_ttl_on_update_secs)

        if cache_max_memory_mb is not None:
            cache_max_memory_mb = int(cache_max_memory_mb)

        if cache_query_ttl_secs is not None:
            cache_query_ttl_secs = int(cache_query_ttl_secs)

        if cache_feedback_ttl_secs is not None:
            cache_feedback_ttl_secs = int(cache_feedback_ttl_secs)

        return super(SystemConfig, cls).__new__(
            cls, cache_ttl_on_update_secs,
            cache_max_memory_mb,
            cache_query_ttl_secs,
            cache_feedback_ttl_secs,
            adhoc_execute)

    @classmethod
    def from_config_items(cls, config_items):
        """
        Takes the output of a RawConfigParser.items call.
        """
        return SystemConfig(**config_items)

    @classmethod
    def from_config_parser(cls, config_parser, section_name):
        items = config_parser.items(section_name) \
            if config_parser.has_section(section_name) else []

        # convert to a dict
        config_dict = {k:v for (k,v) in items}
        return cls.from_config_items(config_dict)

    def set_state_in_config(self, config, section_name):
        for field in self._fields:
            if field in _ALL_SUPPORTED_CONFIGS:
                config.set(section_name, field, getattr(self, field))

    def for_json(self):
        return {k: v for k, v in self._asdict().items() if v is not None}

    def copy_and_update(self, **kwargs):
        # A tuple is immutable.
        # _replace() returns a new tuple with your modifications.
        return self._replace(**kwargs)

def create(name, ec2_config, state_path, license_file_path = None, num_hosts = 1,
           description = None, api_key = None, admin_key = None,
           ssl_credentials = None, cors_origin = '',
           port = _DEFAULT_ADMIN_PORT, admin_ui_port = _DEFAULT_ADMIN_UI_PORT,
           scheme='internet-facing',
           config_file=None):
    """Refer to psclient.create"""
    if not _name_checker.match(name):
        raise ValueError('Predictive Service Name can only contain: a-z, A-Z and hyphens.')
    if len(name) > 32:
        raise ValueError("Predictive Service name can not be longer than 32 characters.")

    if num_hosts > _NODE_LAUNCH_LIMIT:
        raise ValueError("You cannot launch more than %d nodes at one time. " \
                         "If this limit is problematic, please visit " \
                         "https://turi.com/support for support options." % _NODE_LAUNCH_LIMIT)

    # Validate Ec2 Config
    if not isinstance(ec2_config, Ec2Config):
        raise TypeError('Unsupported type given for ec2_config parameter. Must be an Ec2Config object.')

    # Save AWS config
    if(hasattr(ec2_config, 'aws_access_key_id') and hasattr(ec2_config, 'aws_secret_access_key')):
        aws_access_key = ec2_config.aws_access_key_id
        aws_secret_key = ec2_config.aws_secret_access_key
    else:
        raise IOError('No AWS credentials set. Credentials must either be set in the ' \
                          'ec2_config parameter or set globally using ' \
                          'psclient.aws.set_credentials(...).')
    aws_credentials = {
        'aws_access_key_id': aws_access_key,
        'aws_secret_access_key': aws_secret_key
    }

    if _file_util.exists(state_path, aws_credentials):
        raise RuntimeError('Path %s already exists, choose a different path as state path' % state_path)

    # Warn if specified bucket is in different region than specified in env.
    s3_bucket_name, _ = _file_util.parse_s3_path(state_path)

    if license_file_path:
        license_file_path = _os.path.abspath(_os.path.expanduser(license_file_path))
    else:
        license_file_path = _os.path.join(_os.path.expanduser("~"), ".graphlab", "config")
    try:
        _file_util.upload_to_s3(license_file_path, state_path + "/license",
                                aws_credentials = aws_credentials, silent = True)
        region = _file_util.get_s3_bucket_region(s3_bucket_name, aws_credentials)
    except _S3ResponseError as e:
        _logger.error("Unable to connect to state_path's bucket; check your AWS credentials")
        raise

    if region != ec2_config.region:
        _logger.warn("The bucket in your state path is in a different region " \
                     "(%s) than the one specified in your environment (%s). " \
                     "AWS data transfer rates apply. Additionally, upload and " \
                     "download speeds may be slower than expected. If this is " \
                     "not what you intended, abort this operation or " \
                     "terminate the service upon its completion, then be sure " \
                     "that your environment and S3 bucket are in the same " \
                     "region before relaunching." % (region, ec2_config.region))

    conn = _lb_connection(ec2_config.region, **aws_credentials)
    for lb in conn.get_all_load_balancers():
        if lb.name == name:
            raise IOError('There is already a load balancer with that name. Load balancer names' \
                              ' must be unique in their region. Please choose a different name.')

    _logger.info("Launching Predictive Service with %d hosts, as specified by num_hosts parameter"
                 % (num_hosts))

    # Set defaults values, if needed
    if not api_key:
        api_key = str(_random_uuid())
    if not admin_key:
        admin_key = str(_random_uuid())

    # get product key
    product_key = _Ec2PredictiveServiceEnvironment._get_product_key(
                                            license_file_path, aws_credentials)

    result = None
    env = None
    try:
        starttime = _datetime.datetime.now()
        _logger.info("Launching Predictive Service, with name: %s" % name)

        _logger.info("[Step 0/5]: Initializing S3 locations.")

        # Check for invalid names and port
        _check_name_and_port(name, port)

        # Launch the environment.
        env = _Ec2PredictiveServiceEnvironment.launch(name, ec2_config, state_path, num_hosts,
                                                      ssl_credentials, aws_credentials, started=starttime,
                                                      admin_port=port, admin_ui_port=admin_ui_port,
                                                      product_key=product_key, scheme=scheme)

        # Create initial state file to save
        state = _generate_init_state_file(name, env, description, api_key,
                                          admin_key, aws_credentials,
                                          cors_origin, 'enabled',
                                          SystemConfig())

        # Save initial state file with environment
        _save_initial_state(env, state, state_path)

        # create endpoint
        endpoint = 'https://' if ssl_credentials else 'http://'
        endpoint = endpoint + env.load_balancer_dns_name + ":" + str(port)

        _logger.info("[Step 4/5]: Waiting for Load Balancer to put all instances into service.")
        while ((_datetime.datetime.now() - starttime).total_seconds() < _MAX_CREATE_TIMEOUT_SECS):
            # query status, verify all InService
            nodes = env._get_load_balancer_status()
            statuses = []
            for node in nodes:
                statuses.append(node['state'] == 'InService')
            if all(statuses):
                _logger.info("Cluster is fully operational, [%d/%d] instances currently in service." %
                        (statuses.count(True), len(statuses)))
                break
            else:
                _logger.info("Cluster not fully operational yet, [%d/%d] instances currently in service." %
                        (statuses.count(True), len(statuses)))
                _time.sleep(15)
        else:
            _logger.error("Instances failed to be ready within 10 minutes. Tearing down.")
            raise RuntimeError("Cluster configuration not successful in time, timing out.")

        _logger.info("[Step 5/5]: Finalizing Configuration.")

    except Exception as e:
        # any exceptions we should gracefully terminate / tear down what we've created
        _logger.warning("Tearing down Predictive Service due to error launching, %s" % e)

        # default behavior deletes the log files in tear down.
        # To preserve the logs set GRAPHLAB_DEBUG in environment, and the logs will be preserved
        delete_logs = True
        if 'GRAPHLAB_DEBUG' in _os.environ:
            _logger.info("Preserving Log Files for debugging purposes")
            delete_logs = False

        if env:
            env.terminate(delete_logs)

        if result and delete_logs:
            _logger.info('Deleting model data.')
            try:
                _file_util.s3_recursive_delete(state_path, aws_credentials)
            except:
                _logger.error("Could not delete model data. Please manually delete data under: %s" %
                              state_path)

        raise


    # Create the predictive service object.
    ps = _PredictiveService(endpoint, admin_key)

    if not config_file:
        config_file = _os.path.join(
            _os.getcwd(),
            name+'.conf')

    generate_config(config_file, ps)
    _logger.info("Created config file at %s", config_file)
    _logger.info("Contents are:\n%s", open(config_file).read())

    return ps

create.__doc__ = psclient.create.__doc__
