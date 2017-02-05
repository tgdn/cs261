from boto.ec2 import elb as _elb
from boto import ec2 as _ec2
import os
from requests import get, post
from psclient import file_util

# TODO: move all port to one location, and make sure pass the PORT to management server
DEFAULT_ADMIN_PORT = 8088
DEFAULT_ADMIN_UI_PORT = 8888
DEFAULT_QUERY_PORT = 9005
REDIS_PORT = 9006
ENV_TYPE = 'environment type'

import logging as _logging
_logger = _logging.getLogger(__name__)

# disable spurious output from requests library
_logging.getLogger("requests").setLevel(_logging.WARNING)

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
        self._should_verify_certificate = None
        self.port = port
        self.admin_port = port
        self.admin_ui_port = admin_ui_port
        self.aws_credentials = aws_credentials

    def list_file(self, remote_path):
        if file_util.is_local_path(remote_path):
            if os.path.isdir(remote_path):
                return os.listdir(remote_path)
            else:
                return []

        elif file_util.is_s3_path(remote_path):
            files = file_util.list_s3(remote_path, aws_credentials = self.aws_credentials)
        else:
            files = file_util.list_hdfs(remote_path)

        if not files or len(files) == 0:
            return []

        return_value = set()

        # strip the relative path
        for f in files:
            rel_path = os.path.relpath(f['path'], remote_path)

            # Only show first level children
            if rel_path != '.' and rel_path != '..':
                # In windows, the relpath could replace the separator with '\'
                return_value.add(rel_path.replace('\\', '/').split('/')[0])

        return_value = list(return_value)
        return_value.sort()
        return return_value

class LocalPredictiveServiceEnvironment(PredictiveServiceEnvironment):
    def __init__(
            self,
            log_path,
            load_balancer_dns_name,
            aws_credentials=None,
            num_hosts = 1,
            redis_manager = None,
            web_server = None,
            graphlab_service = None,
            admin_port = DEFAULT_ADMIN_PORT,
            admin_ui_port = DEFAULT_ADMIN_UI_PORT,
            port = DEFAULT_QUERY_PORT, **kwargs):

        PredictiveServiceEnvironment.__init__(self, admin_port, admin_ui_port, port, aws_credentials)
        self.load_balancer_dns_name = load_balancer_dns_name
        self.log_path = log_path
        self.num_hosts = int(num_hosts)
        self.redis_manager = redis_manager
        self.web_server = web_server
        self.graphlab_service = graphlab_service
        self.certificate_is_self_signed = True

    def get_all_hosts(self):
        return [{
                'id': 'localhost:%s' % port,
                'dns_name': self.load_balancer_dns_name,
                'state': 'InService',
                'ip_address': '127.0.0.1',
                'reason': None,
                'private_ip_address': '127.0.0.1',
                'redis_port': port}
                for port in range(9006, 9006 + self.num_hosts)]

class DockerPredictiveServiceEnvironment(PredictiveServiceEnvironment):

    def __init__(self,
            load_balancer_dns_name,
            log_path,
            certificate_name = None,
            certificate_is_self_signed = True,
            aws_credentials = None,
            admin_port = DEFAULT_ADMIN_PORT,
            admin_ui_port = DEFAULT_ADMIN_UI_PORT,
            port = DEFAULT_QUERY_PORT,
            load_balancer_stats_port = DEFAULT_QUERY_PORT - 1,
            metrics_server = None):

        PredictiveServiceEnvironment.__init__(self, admin_port, admin_ui_port, port, aws_credentials)
        self.load_balancer_dns_name = load_balancer_dns_name
        self.log_path = log_path
        self.ctl_port = load_balancer_stats_port
        self.metrics_server = metrics_server
        self.certificate_name = certificate_name
        self.certificate_is_self_signed = certificate_is_self_signed

    def get_all_hosts(self):
        assert len(self._get_all_hosts()) == 1
        host = self._get_all_hosts()[0]
        dns_name, port = host['svname'].split(':')
        smalldict = {}
        smalldict['id'] = dns_name
        smalldict['dns_name'] = dns_name
        smalldict['port'] = port
        smalldict['state'] = host['status']
        smalldict['reason'] = host['last_chk']
        smalldict['redis_port'] = REDIS_PORT
        smalldict['private_ip_address'] = dns_name
        return [smalldict]

    def _get(self, url, params=None):
        if not self._should_verify_certificate:
            self._should_verify_certificate = self.certificate_name and len(self.certificate_name) > 0 and not self.certificate_is_self_signed

        return get(url=url, params=params, verify=self._should_verify_certificate, timeout=10)

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

class Ec2PredictiveServiceEnvironment(PredictiveServiceEnvironment):

    def __init__(self,
            load_balancer_dns_name,
            region,
            log_path,
            certificate_name,
            certificate_is_self_signed,
            aws_credentials,
            admin_port = DEFAULT_ADMIN_PORT,
            admin_ui_port = DEFAULT_ADMIN_UI_PORT,
            port = DEFAULT_QUERY_PORT):

        PredictiveServiceEnvironment.__init__(
                        self, admin_port, admin_ui_port, port, aws_credentials)

        self.region = region
        self.load_balancer_dns_name = load_balancer_dns_name
        self.log_path = log_path
        self.certificate_name = certificate_name
        self.certificate_is_self_signed = certificate_is_self_signed

    def _get_load_balancer_handle(self):
        try:
            conn = _elb.connect_to_region(self.region, **self.aws_credentials)
            for i in conn.get_all_load_balancers():
                if i.dns_name == self.load_balancer_dns_name:
                    return i
        except Exception as e:
            _logger.error("Unable to connect to ELB with name '%s' in region '%s', error: %s"
                          % (self.load_balancer_dns_name, self.region, e))

        raise Exception("Cannot find load balancer with name '%s' in region '%s'."
                        % (self.load_balancer_dns_name, self.region))

    def get_lb_host_info(self):
        '''
        Get host status from load balancer point of view
        '''
        result = {}
        load_balancer = self._get_load_balancer_handle()

        for host in load_balancer.get_instance_health():
            host_id = host.instance_id
            host_instance = self._get_ec2_instances([host_id], self.region,
                                               aws_credentials = self.aws_credentials)
            private_ip_address = host_instance[0].private_ip_address

            smalldict = {}
            smalldict['id'] = host_id
            smalldict['ip_address'] = host_instance[0].ip_address
            smalldict['state'] = host.state
            smalldict['reason'] = host.reason_code
            result[private_ip_address] = smalldict
        return result

    def _get_ec2_instances(self, ec2_id_list, region, aws_credentials = {}):
        '''
        Get all instances information
        '''
        conn = _ec2.connect_to_region(region, **aws_credentials)
        response = conn.get_all_instances(instance_ids = ec2_id_list)
        results = []
        for reservation in response:
            results += reservation.instances
        return results

    def get_all_hosts(self):
        '''
        Get all host information, returns the information regarding dns_name,
        private_ip, redis_port, etc.
        '''
        load_balancer_handle = self._get_load_balancer_handle()

        host_ids = [i.id for i in load_balancer_handle.instances]

        if len(host_ids) == 0:
            return []

        result = []
        for host in load_balancer_handle.get_instance_health():
            host_id = host.instance_id
            host_instance = self._get_ec2_instances([host_id], self.region,
                                               aws_credentials = self.aws_credentials)

            smalldict = {}
            smalldict['id'] = host_instance[0].id
            smalldict['dns_name'] = host_id
            smalldict['ip_address'] = host_instance[0].ip_address
            smalldict['state'] = host.state
            smalldict['reason'] = host.reason_code
            smalldict['private_ip_address'] = host_instance[0].private_ip_address
            smalldict['redis_port'] = REDIS_PORT
            result.append(smalldict)

        return result
