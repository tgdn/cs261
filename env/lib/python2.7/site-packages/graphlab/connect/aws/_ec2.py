"""@package graphlab.connect.aws.ec2

This module makes it easy to have a GraphLab server running on an EC2 instance.
"""

'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''

import json
import logging
import os
import time

import boto.ec2
import boto.vpc
from boto.exception import EC2ResponseError

import graphlab
import graphlab.product_key

from graphlab.util.config import DEFAULT_CONFIG as config
from .block_device_mapping import get_block_device_mapping

# Do not remove, used by other components to get credentials
from graphlab.util import _get_aws_credentials as get_credentials

import decorator

import sys
if sys.version_info.major == 2:
    import ConfigParser
    from urllib2 import urlopen
else:
    import configparser as ConfigParser
    import urllib.request as urlopen


CONFIG_SECTION = 'AWS'
DEFAULT_CIDR_RULE = '0.0.0.0/0'  # Open to the entire internet.
DEFAULT_INSTANCE_TYPE = 'm3.xlarge'
JSON_BLOB_PATH_FORMAT = "/gl_api/api/v1/cloud/aws/instances/%s?client_version=%s&region=%s&product_key=%s"
TURI_DISTRIBUTED_NAME = 'Turi_Distributed'
VALID_REGIONS = [r.name for r in boto.ec2.get_regions('ec2')]

# Port 9000 & 9001 - used when running in server mode: only server running with external client.
# Port 9002 & 9003 - used for metrics (deprecated).
# Port 9004 - old commander daemon for jobs/tasks (deprecated).
# Port 9005 - (was used for predictive service/web service. It is the default port used if user does not specify any port))
# Port 9006 - used by predictive service (reddis).
# Port 9007 - old worker daemon for jobs/tasks (deprecated).
# Port 9000 - 9200 - new commander worker, used by Turi Distributed
# Port 19006 - used by predictive service (reddis).
TURI_DISTRIBUTED_PORT_MIN_NUM = 9000
TURI_DISTRIBUTED_PORT_MAX_NUM = 9200
REDIS_PORT = 9006
ADDITIONAL_REDIS_PORT = 19006

# Swallow all boto logging, except critical issues
logging.getLogger('boto').setLevel(logging.CRITICAL)

__LOGGER__ = logging.getLogger(__name__)

class LicenseValidationException(Exception):
    pass

class _ProductType():
    TuriDistributed = 0

class _Ec2Instance:
    def __init__(self, ip_address, private_ip_address, instance_id, instance, region):
        self.ip_address = ip_address
        self.private_ip_address = private_ip_address
        self.instance_id = instance_id
        self.instance = instance
        self.region = region

    def stop(self):
        _stop_instances([self.instance_id], self.region)


def _get_region_from_config(config_path = (os.path.join(os.path.expanduser("~"), ".graphlab", "config"))):
    result = None
    if (os.path.isfile(config_path)):
        config = ConfigParser.ConfigParser()
        config.read(config_path)
        if(config.has_section(CONFIG_SECTION) and config.has_option(CONFIG_SECTION, 'region')):
            result = config.get(CONFIG_SECTION, 'region')
    return result


def set_credentials(access_key_id, secret_access_key):
    """
    Sets the AWS credential environment variables.

    Helper function to set the following environment variables: AWS_ACCESS_KEY_ID and
    AWS_SECRET_ACCESS_KEY. You are also free to set these environment variables directly.

    Parameters
    ----------
    access_key_id : str
        Value for the AWS_ACCESS_KEY_ID environment variable.

    secret_access_key : str
        Value for the AWS_SECRET_ACCESS_KEY environment variable.

    See Also
    --------
    get_credentials

    Examples
    --------
    >>> graphlab.aws.set_credentials('RBZH792CTQPP7T435BGQ', '7x2hMqplWsLpU/qQCN6xAPKcmWo46TlPJXYTvKcv')

    """
    os.environ['AWS_ACCESS_KEY_ID'] = str(access_key_id)
    os.environ['AWS_SECRET_ACCESS_KEY'] = str(secret_access_key)


# Decorator which catch the exception and output to log error.
@decorator.decorator
def __catch_and_log__(func, *args, **kargs):
        try:
            return func(*args, **kargs)
        except Exception as error:
            logging.getLogger(__name__).error(error)

@__catch_and_log__
def list_instances(region='us-west-2'):
    """
    Instance IDs for all active GraphLab EC2 hosts. Returns instance IDs for all
    active GraphLab EC2 hosts (for the specified region), not just the instance
    ID associated with the current python session.

    Parameters
    ----------
    region : string, optional
        The `AWS region
        <http://docs.aws.amazon.com/general/latest/gr/rande.html#ec2_region>`_
        in which to list instances. Default is 'us-west-2'.

    Returns
    -------
    out : list[string]
        A list of AWS instance IDs corresponding to active instances in the
        specified region.

    Examples
    --------
    .. sourcecode:: python

        # No instances running in 'us-west-2'.
        >>> graphlab.aws.list_instances()
        []

        # Two insances running in 'us-east-1'.
        >>> graphlab.aws.list_instances(region = 'us-east-1')
        [u'i-8df559a7', u'i-ed2aa9c1']
    """

    __LOGGER__.info("Listing instances in %s." % region)
    aws_connection = boto.ec2.connect_to_region(region)
    recent_ids = [i.res_id for i in aws_connection.get_all_tags(filters={'tag:GraphLab': ''})]
    active_instnaces = aws_connection.get_all_instance_status(instance_ids=recent_ids)
    return [i.id for i in active_instnaces]


def _stop_instances(instance_id_list, region, credentials = {}):
    __LOGGER__.info("Terminating EC2 host(s) %s in %s" % (instance_id_list, region))

    # Hopefully we will not need this string
    ERROR_MSG = 'Invalid response from EC2. Unable to terminate host(s). Please terminate using the ' \
        'AWS EC2 Console.'

    try:
        # Shut the host(s) down.
        conn = boto.ec2.connect_to_region(region, **credentials)
        response = conn.terminate_instances(instance_ids = instance_id_list)

        # Sanity check response.
        if(not(len(response) == len(instance_id_list))):
            raise Exception(ERROR_MSG)

         # it is possible the instances are still being terminated, but we do
         # not need to wait for that now.
        return
    except:
        raise Exception(ERROR_MSG)

def _authorize_for_security_group(security_group, ip_protocol, from_port, to_port,
                                  cidr_ip=None, src_group=None):
    # Note that this will throw an exception if this particular set of ports was already authorized in the security group.
    try:
        security_group.authorize(ip_protocol, from_port, to_port, cidr_ip, src_group)
    except EC2ResponseError as e:
        if not "<Code>InvalidPermission.Duplicate</Code>" in str(e):
            raise e

def _authorize_for_turi_distributed(security_group, cidr_ip):
    # Allow all traffic between hosts in same security group.
    _authorize_for_security_group(security_group = security_group,
               ip_protocol = 'tcp',
               from_port = 0,
               to_port = 65535,
               src_group = security_group)

    # Ports open externally.
    _authorize_for_security_group(security_group = security_group,
                                  ip_protocol = 'tcp',
                                  from_port = TURI_DISTRIBUTED_PORT_MIN_NUM,
                                  to_port = TURI_DISTRIBUTED_PORT_MAX_NUM,
                                  cidr_ip = cidr_ip)

    _authorize_for_security_group(security_group = security_group,
               ip_protocol = 'tcp',
               from_port = 443,
               to_port = 443,
               cidr_ip = cidr_ip)
    _authorize_for_security_group(security_group = security_group,
               ip_protocol = 'tcp',
               from_port = 80,
               to_port = 80,
               cidr_ip = cidr_ip)


# The user might specify various combination of security_group_name, security_group_id
# and subnet_id, so we have to resolve them.
# From EC2 documentation,
# "When you launch an EC2 instance without specifying a subnet,
# it's automatically launched into a default subnet in your default VPC
# To launch an EC2 instance into a specific default subnet in your default VPC,
# specify its subnet ID"
# If a security_group_name is specified, it should be for a subnet in a default VPC.
# For other VPCs, a security_group_id needs to be specified.
def _get_security_group_config(product_type, region, credentials, security_group_name,
                               security_group_id, subnet_id):
    if product_type is _ProductType.TuriDistributed:
        security_group_default_name = TURI_DISTRIBUTED_NAME

    try:
        conn = boto.vpc.connect_to_region(region, **credentials)
        #We do not allow both security_group_name and security_group_id to be specified.
        if security_group_name is not None and security_group_id is not None:
            raise Exception("Specify either security_group_name or security_group_id but not both.")

        elif security_group_name is None and security_group_id is None:
            if subnet_id is None:
            # If we have no information at all, we look for existing security group with
            # the security_group_default_name. If it exists, we need to know what is the corresponding
            # subnet_id. If not, we create one without specifying any VPC.
            # AWS will use the default VPC.
                security_group = None
                for sg in conn.get_all_security_groups():
                    if(sg.name == security_group_default_name):
                        security_group = sg
                        subnets = conn.get_all_subnets(filters={'vpc_id': security_group.vpc_id})
                        if len(subnets) != 0:
                            subnet_id = subnets[0].id
                if security_group is None:
                    security_group = conn.create_security_group(security_group_default_name,
                        'Only open ports needed for ' + security_group_default_name)

            else:
                # If a subnet is specified but nothing about security group is specified,
                # we look for existing security group with the name security_group_default_name. If it exists,
                # we use it. If not, we create a new security_group in this the subnet.
                subnets = conn.get_all_subnets(subnet_ids=[subnet_id])
                if len(subnets) == 0:
                    raise RuntimeError('Cannot find subnet with id %s' % subnet_id)
                security_group = None
                for sg in conn.get_all_security_groups():
                    if(sg.name == security_group_default_name and sg.vpc_id == subnets[0].vpc_id):
                        security_group = sg
                        break
                if security_group is None:
                    security_group = conn.create_security_group(security_group_default_name,
                        'Only open ports needed for ' + security_group_default_name, vpc_id = subnets[0].vpc_id)

        elif security_group_id is not None:
            security_groups = conn.get_all_security_groups(group_ids=[security_group_id])
            security_group = security_groups[0]
            if subnet_id is None:
                # If user gives us a security_group_id that is meant for a nondefault vpc,
                # but do not provide us with a subnet_id, we need to figure out the what is
                # the id for the subnet inside the vpc for the corresponding security_group_id.
                # Otherwise, AWS would assume we want to launch it into a default subnet
                # inside a default VPC, which would give us an error because the security_group_id
                # is not meant for this default VPC.
                # This will also work for the case that the security_group_id is meant
                # for a default vpc.
                subnets = conn.get_all_subnets(filters={'vpc_id': security_group.vpc_id})
                if len(subnets) != 0:
                    subnet_id = subnets[0].id

        elif security_group_name is not None:
            # If security_group_name is specified, we still need to check whether it already exists.
            # If it exists, use it.
            # If it does not exist, we create a new one that has security_group_name as the name.
            security_group = None
            security_groups = conn.get_all_security_groups()
            for sg in conn.get_all_security_groups():
                if(security_group_name == sg.name):
                    security_group = sg
                    # If there is an existing security group with that name, the security_group might
                    # be from a non-default VPC and we need to know which subnet the security_group
                    # is from.
                    if subnet_id is None:
                        subnets = conn.get_all_subnets(filters={'vpc_id': security_group.vpc_id})
                        if len(subnets) != 0:
                            subnet_id = subnets[0].id
                    # subnet_id is not None case:
                    #   Note: We do not check whether or not security_group_name
                    #       and subnet_id are in the same vpc, we defer to boto's
                    #       launch_instance to do that
                    break
            if security_group is None:
                if subnet_id is None:
                    security_group = conn.create_security_group(security_group_name,
                        'Only open ports needed for ' + security_group_default_name)
                else:
                    subnets = conn.get_all_subnets(subnet_ids=[subnet_id])
                    security_group = conn.create_security_group(security_group_name,
                        'Only open ports needed for ' + security_group_default_name,
                        vpc_id = subnets[0].vpc_id)
        return security_group, subnet_id

    except boto.exception.EC2ResponseError as e:
        raise Exception('EC2 response error. Exception: %s' % e)
    except boto.exception.NoAuthHandlerFound as e:
        # Most likely cause is env vars aren't set.
        raise Exception('AWS configuration not found. Please configure your AWS credentials '
                'using graphlab.aws.set_credentials or by setting AWS_ACCESS_KEY_ID '
                'and AWS_SECRET_ACCESS_KEY environment variables')

def _setup_security_group(region, CIDR_rule, security_group_name,
                          credentials, additional_port_to_open,
                          product_type,
                          subnet_id, security_group_id):

    security_group, subnet_id = _get_security_group_config(
        product_type, region, credentials,
        security_group_name, security_group_id, subnet_id)

    if CIDR_rule is None:
        CIDR_rule = DEFAULT_CIDR_RULE

    if product_type is _ProductType.TuriDistributed:
        _authorize_for_turi_distributed(security_group = security_group,
                                        cidr_ip = CIDR_rule)
    else:
        raise Exception("Product Type (Turi Distributed) is not specified.")

    return [security_group, subnet_id]

def _ec2_factory(instance_type, region=None, availability_zone=None,
                 CIDR_rule=None, security_group_name=None, tags=None,
                 user_data = {}, credentials = {}, ami_service_parameters = {},
                 num_hosts = 1, additional_port_to_open = None,
                 product_type = None,
                 subnet_id = None, security_group_id = None):

    '''
    This function does everything necessary to bring up EC2 host(s): create a security group (if
    nessary), determine arguments to start up the EC2 instance (i.e. AMI id and user data),
    actually start up the EC2 instance, wait for it, and applies AWS tags.
    '''
    from graphlab.connect.main import get_unity, is_connected, ENGINE_START_ERROR_MESSAGE
    from graphlab.product_key import get_product_key

    # Before launching EC2 instances we want to make sure the product key is valid. So make sure
    # the server has started.
    get_unity()
    assert is_connected(), ENGINE_START_ERROR_MESSAGE
    product_key = get_product_key()

    # Set default values for parameters.
    if(region is None):
        region = _get_region_from_config()
        if(region is None):
            region = 'us-west-2'
        else:
            __LOGGER__.info('Read region from config file.')

    if (region not in VALID_REGIONS):
        raise Exception("%s is not a valid region." % region)

    security_group, subnet_id = _setup_security_group(region = region, CIDR_rule = CIDR_rule,
        security_group_name = security_group_name, credentials = credentials,
        additional_port_to_open = additional_port_to_open, product_type = product_type,
        subnet_id = subnet_id, security_group_id = security_group_id)

    if ('GRAPHLAB_TEST_AMI_ID' in os.environ and 'GRAPHLAB_TEST_ENGINE_URL' in os.environ and 'GRAPHLAB_TEST_HASH_KEY' in os.environ):
        # unit-test mode, don't involve webservice to retrieve AMI, instead use environment variables
        ami_id = os.environ['GRAPHLAB_TEST_AMI_ID']
        engine_url = os.environ['GRAPHLAB_TEST_ENGINE_URL']
        __LOGGER__.info("UNIT mode, using AMI: '%s' and engine url: '%s' when launching EC2 instance." % (ami_id, engine_url))
        json_blob = json.loads('{}')
        json_blob['ami_id'] = ami_id
        json_blob['engine_url'] = engine_url
        json_blob['hash_key'] = os.environ['GRAPHLAB_TEST_HASH_KEY']
    else:
        # Get the info to start a EC2 from the GraphLab Server
        json_blob_path = JSON_BLOB_PATH_FORMAT % (instance_type, graphlab.version, region, product_key)
        for (param_name, param_value) in ami_service_parameters.items():
            json_blob_path += "&%s=%s" % (str(param_name), str(param_value))
        json_blob_url = config.graphlab_server + json_blob_path

        try:
            # set specific timeout for this web service request, lots of time spent in SSL negotiation
            # for staging server allows a little more time
            timeout_in_seconds = 10 if config.mode == 'PROD' else 60
            graphlab_server_response = urlopen(json_blob_url, timeout=timeout_in_seconds)
            json_blob = json.loads(graphlab_server_response.read().decode('utf-8'))
        except:
            raise Exception('Unable to successfully retrieve correct EC2 image to launch for this '
                    'version. This could be a temporary problem. Please try again in a few '
                    'minutes.')
        __LOGGER__.debug("web service return: %s" % json_blob)

        if json_blob.get('error'):
            raise LicenseValidationException(json_blob.get('error'))

    if 'ami_id' not in json_blob or json_blob['ami_id'] is None:
        raise Exception("Unable to successfully retrieve correct EC2 image to launch. Please try "
                "again later. Error received:'%s'"
                % json_blob.get('message'))
    ami_id = json_blob['ami_id']

    # Add json_blob to user_data and set the product key and hash key
    user_data.update(json_blob)
    user_data['product_key'] = product_key

    user_data['hash_key'] = json_blob.get('hash_key', 'NO_HASH_VALUE')

    # Check for empty os_url
    if user_data.get('os_url') is None or len(user_data.get('os_url')) == 0:
        user_data['os_url'] = 'NO_OS_URL'

    # Check for testing override of os_url param.
    if ('GRAPHLAB_TEST_OS_URL' in os.environ):
        user_data['os_url'] = os.environ['GRAPHLAB_TEST_OS_URL']

    run_instances_args =  {
            'security_group_ids' : [ security_group.id ],
            'user_data' : json.dumps(user_data),
            'instance_type' : instance_type,
            'placement' : availability_zone,
            'subnet_id' : subnet_id
    }

    if num_hosts != 1:
        run_instances_args['min_count'] = num_hosts
        run_instances_args['max_count'] = num_hosts

    if 'GRAPHLAB_TEST_EC2_KEYPAIR' in os.environ:
        keypair = os.environ['GRAPHLAB_TEST_EC2_KEYPAIR']
        __LOGGER__.info("Using keypair: '%s' when launching EC2 instance" % (keypair))
        run_instances_args['key_name'] = keypair

    run_instances_args['block_device_map'] = get_block_device_mapping(instance_type)

    # Actually launch the EC2 instance(s) and wait for them to start running.
    instances = None
    try:
        conn = boto.vpc.connect_to_region(region, **credentials)
        response = conn.run_instances(ami_id, **run_instances_args)
        instances = response.instances
        if(len(response.instances) != num_hosts):
            raise Exception

        # Report
        for i in instances:
            __LOGGER__.info("Launching an %s instance in the %s availability zone, with id: %s."
                    " You will be responsible for the cost of this instance."
                    % (i.instance_type, i.placement, i.id))

        # Wait for all host(s) to say they're done starting up.
        while True:
            try:
                for i in instances:
                    # Rarely an instance can a reach temp state before going into pending. We check for
                    # 'running' right away to make unit tests work.
                    while not i.update() in ['pending', 'running', 'failed']:
                        time.sleep(1)
                    while i.update() == 'pending':
                        time.sleep(1)
                    if i.update() == 'failed':
                        raise RuntimeError("Instance %s startup failed" % i.id)
                break
            except EC2ResponseError as e:
                # EC2 is eventual consistence so sometimes it complains that it
                # cannot find the instance, in that case, we will retry
                __LOGGER__.debug("Ignoring EC2ResponseError: %s" % e.message)

        # Add tags to this instance(s).
        if(tags is None):
            tags = {}
        if product_type is _ProductType.TuriDistributed:
            security_group_default_name = TURI_DISTRIBUTED_NAME
        tags[security_group_default_name] = ''
        for i in instances:
            conn.create_tags(i.id, tags)

        results = []
        for i in instances:
            results.append(_Ec2Instance(i.ip_address, i.private_ip_address, i.id, i, region))

        if num_hosts == 1:
            # for backward compatibility
            return [results[0], security_group, subnet_id]
        return [results, security_group, subnet_id]

    except Exception as e:
        if instances:
            _stop_instances([i.id for i in instances] , region)
        raise Exception("Unable to launch EC2 instance: '%s'. Please check AWS Console to make"
                " sure any EC2 instances launched have been terminated." % e)

def _get_ec2_instances(ec2_id_list, region, aws_credentials = {}):
    conn = boto.ec2.connect_to_region(region, **aws_credentials)
    response = conn.get_all_instances(instance_ids = ec2_id_list)

    results = []
    for reservation in response:
        results += reservation.instances

    return results
