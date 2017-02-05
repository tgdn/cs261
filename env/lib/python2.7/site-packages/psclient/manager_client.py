"""Predictive Service Manager Client Code"""
from .rest import (
    RESTObject,
    RESTProperty,
    enum,
)

from datetime import datetime

class SystemCacheConfiguration(RESTObject):
    """The system-side cache configuration.

    Attributes
    ----------
    global_cache_state : string
        "enabled" or "disabled"

    cache_max_memory_mb : int
        The maximum memory the cache will use in mb.

    cache_ttl_on_update_secs : float
        How long a particular cache entry will live after an update in
        seconds.

    cache_query_ttl_secs : float
        How long a particular cache entry will live after being queried in
        seconds.

    cache_feedback_ttl_secs : float
        How long a particular cache entry will live after receiving feedback
        in seconds.
    """
    global_cache_state = RESTProperty(enum("enabled", "disabled"))
    cache_max_memory_mb = RESTProperty(int)
    cache_ttl_on_update_secs = RESTProperty(int)
    cache_query_ttl_secs = RESTProperty(int)
    cache_feedback_ttl_secs = RESTProperty(int)

class ScaleFactorConfiguration(RESTObject):
    """The system-wide scale factor configuration.

    Attributes
    ----------
    scale_factor : int
        The scale factor, which should be a positive integer.
    """
    scale_factor = RESTProperty(int)

class EndpointCacheConfiguration(RESTObject):
    """Cache configuration for a single endpoint.

    Attributes
    ----------

    cache_state : str
        "enabled" or "disabled"
    """

    cache_state = RESTProperty(enum("enabled", "disabled"))

class AdhocExecuteConfiguration(RESTObject):
    """The system-wide adhoc execute configuration.

    Attributes
    ----------
    adhoc_execute : str
        "enable" or "disable", depending on whether the adhoc execute should
        be allowed.

        Note: The client must either use an admin key or an api key with
        adhoc_execute enabled in order to execute adhoc queries, in addition to
        this being enabled.
    """
    adhoc_execute = RESTProperty(enum("enable", "disable"))

class CORSConfiguration(RESTObject):
    """The system-wide CORS configuration.

    Attributes
    ----------
    cors_origin : str
        The CORS origin for the system.
    """
    cors_origin = RESTProperty(str)

class Key(RESTObject):
    """Keys are used to limit access to your Predictive Service.

    Admin keys (type='admin') can modify the service and query any endpoint.

    API keys (type='api') can only query endpoints listed in it's endpoints
    property.

    Monitor keys (type='monitor') can only query some aspects of the service
    related to metrics and logs and cannot change the service. They also
    cannot query the service.
    """
    id = RESTProperty(str, doc="str: The unique key id. This could be randomly"
        " generated.")
    type = RESTProperty(enum("admin", "api", "monitor"), doc="str: The type of"
        " key: 'admin', 'api', or 'monitor'")
    endpoints = RESTProperty(list, doc="list of str: A list of endpoints the api key will"
        " work for. Only for api keys.")

    def __init__(self, id=None, type=None, endpoints=None):
        if type == 'api' and endpoints is None:
            endpoints = []

        super(Key, self).__init__(
            id=id,
            type=type,
            endpoints=endpoints)

def from_epoch(value):
    if isinstance(value, datetime):
        return value
    else:
        return datetime.utcfromtimestamp(value)

def to_epoch(value):
    return (value - datetime(1970, 1, 1)).total_seconds()


class Endpoint(RESTObject):
    """Represents an endpoint.

    Note that not every attribute is returned as part of the GET.

    Attributes
    ----------

    name : str

        The name of the endpoint. Valid names include ``[a-zA-Z0-9_\- ]+``

    type : str

        The type of endpoint. The types include "alias", "model", "policy"

    version : int

        The version of this endpoint. Initial versions have version on 1. New
        versions increment this by 1.

    description : str

        A human-readable description of the endpoint.

    dependencies: list

        A list of endpoints that this endpoint depends on.

    methods : list

        ???

    """
    name = RESTProperty(str)
    type = RESTProperty(str)
    version = RESTProperty(int)
    description = RESTProperty(str)
    dependencies = RESTProperty(list)
    methods = RESTProperty(list)
    creation_time = RESTProperty(datetime, from_epoch, to_epoch)
    last_modified_time = RESTProperty(datetime, from_epoch, to_epoch)
    evaluator = RESTProperty(str)
    schema_version = RESTProperty(int)

    def __new__(cls, **kwargs):
        """Dispatch to the appropriate class."""
        cls = {
            'alias': AliasEndpoint,
            'model': ModelEndpoint,
            'policy': PolicyEndpoint,
        }[kwargs['type']]

        return object.__new__(cls, **kwargs)

    def __eq__(self, other):
        return self.name == other.name and \
            self.type == other.type and \
            self.version == other.version and \
            self.description == other.description and \
            self.dependencies == other.dependencies and \
            self.methods == other.methods and \
            self.evaluator == other.evaluator and \
            self.schema_version == other.schema_version

class ModelEndpoint(Endpoint):
    """Represents a model endpoint.

    src_path : str

        The S3, HDFS, or local file path to the source of this model.

    required_files : str

        The S3, HDFS, or local file path to the directory containing the
        required files.

    required_packages : str

        The S3, HDFS, or local file path to the directory containting the
        required packages.

    """
    src_path = RESTProperty(str)
    required_files = RESTProperty(list)
    required_packages = RESTProperty(list)
    required_packages_dst_path = RESTProperty(str)

    def __init__(self, **kwargs):
        super(ModelEndpoint, self).__init__(**kwargs)
        self.type = 'model'

    def __eq__(self, other):
        return super(ModelEndpoint, self).__eq__(other) and \
            self.required_files==other.required_files and \
            self.required_packages == other.required_packages

class AliasEndpoint(Endpoint):
    """Represents an alias Endpoint.

    target : str

        The endpoint that this is an alias for.

    """
    target = RESTProperty(str)

    def __init__(self, **kwargs):
        super(AliasEndpoint, self).__init__(**kwargs)
        self.type = 'alias'

class PolicyEndpoint(ModelEndpoint):
    """Represents a policy endpoint.
    """

    def __init__(self, **kwargs):
        super(PolicyEndpoint, self).__init__(**kwargs)
        self.type = 'policy'

class ManagerServiceClient(object):
    """The client for the PS Manager Service."""

    def __init__(self, service_client):
        self.service_client = service_client

    def get_query_url(self):
        """Returns the query API URL. You can use that to query the endpoints
        with a predictive service client.

        Returns
        -------

        dict

            Keys are:

            * query_url : The query URL.
        """
        return self.get_service_info().get('query_url')

    def get_endpoint_upload_destination(self):
        """Returns a dict representing where endpoint data should be uploaded.

        Returns
        -------

        dict

            Keys include:

            * path: The path. If this starts with "s3://" then it is an S3
              path. If this starts with "hdfs://" then it is an HDFS path.
              Otherwise, it is a local file path.

        Note: In the future, other paths and parameters may be supported.

        Note: At this time, the response should not change over time.
        """
        return self.service_client.GET(
            'configurations/endpoint_upload_destination')

    def get_endpoints(self, type=None):
        """Returns endpoints from the management API.

        Parameters
        ----------

        type : str

            The type of endpoint to return. None will include all endpoints.
            Other options are 'model', 'policy', or 'alias'.
        """
        result = {}
        for name, attrs in self.service_client.GET(
                'endpoints',
                {'type':type}).items():
            endpoint = Endpoint.from_json(attrs)
            endpoint.name = name
            result[name] = endpoint
        return result

    def get_endpoint(self, endpoint_name):
        """Returns an endpoints from the management API given its name.

        Parameters
        ----------

        endpoint_name : str

            The name of the endpoint.
        """
        ((name, attrs),) = self.service_client.GET('endpoints/'+endpoint_name).items()
        endpoint = Endpoint.from_json(attrs)
        endpoint.name = name
        return endpoint

    def add_endpoint(self, endpoint):
        """Adds an endpoint through the management API.

        Parameters
        ----------

        endpoint : Endpoint
        """
        return self.service_client.POST('endpoints',
                endpoint.to_json()
            )

    def set_endpoint(self, endpoint):
        """Updates an endpoint through the management API.

        Parameters
        ----------

        endpoint : Endpoint

            The endpoint to update.
        """
        return self.service_client.PUT('endpoints/'+endpoint.name,
                endpoint.to_json())

    def remove_endpoint(self, endpoint_name):
        """Deletes an endpoint through the management API.

        Parameters
        ----------

        endpoint_name : str

            The endpoint to delete.
        """
        self.service_client.DELETE('endpoints/'+endpoint_name)

    def get_service_info(self):
        """Returns the service info of the Predictive Service.

        This includes environment info, and state path.

        Returns
        -------
        Dict
            A dictionary contains the environment info and state path.
        """
        return self.service_client.GET('info')

    def get_status(self):
        """Returns the status of the nodes in the predictive service.

        Returns
        -------

        Status
            See the example below for the response format.

        Example
        -------

            >>> from pprint import pprint
            >>> pprint(msc.get_status())
            {u'ec2-52-36-213-52.us-west-2.compute.amazonaws.com': {u'cache': {u'healthy': True,
                                                                              u'num_keys': 1,
                                                                              u'type': u'local'},
                                                                   u'dns_name': u'ec2-52-36-213-52.us-west-2.compute.amazonaws.com',
                                                                   u'graphlab_service_status': {u'ip-172-31-16-238:10000': {u'reason': None,
                                                                                                                            u'status': u'healthy'}},
                                                                   u'id': u'i-159ea1cd',
                                                                   u'models': [{u'add': {u'ip-172-31-16-238:10000': {u'cache_enabled': True,
                                                                                                                     u'reason': None,
                                                                                                                     u'status': u'LoadSuccessful',
                                                                                                                     u'type': u'model',
                                                                                                                     u'version': 1}}}],
                                                                   u'reason': u'N/A',
                                                                   u'state': u'InService',
                                                                   u'system': {u'cpu_count': 4,
                                                                               u'cpu_usage': [0.2,
                                                                                              0.1,
                                                                                              0.1,
                                                                                              0.1],
                                                                               u'disk_usage': {u'root': {u'free': 33237012480,
                                                                                                         u'percent': 10.7,
                                                                                                         u'total': 39490912256,
                                                                                                         u'used': 4241055744},
                                                                                               u'tmp': {u'free': 33237012480,
                                                                                                        u'percent': 10.7,
                                                                                                        u'total': 39490912256,
                                                                                                        u'used': 4241055744}},
                                                                               u'memory': {u'active': 1327382528,
                                                                                           u'available': 14820085760,
                                                                                           u'buffers': 274927616,
                                                                                           u'cached': 4492677120,
                                                                                           u'free': 10052481024,
                                                                                           u'inactive': 3838939136,
                                                                                           u'percent': 6.0,
                                                                                           u'total': 15770570752,
                                                                                           u'used': 5718089728}}}}

        """
        return self.service_client.GET('status')

    def get_status_of_nodes(self):
        """Returns a list of dicts representing the status of each node.

        Returns
        -------

        list of dicts
            (What are the keys and values of the dicts?)
        """
        return self.service_client.GET('status/nodes')

    def get_custom_packages(self):
        """Returns the list of custom packages.

        Returns
        -------

        list of str
        """
        return self.service_client.GET('custom_packages')

    def get_metrics_url(self):
        """Returns the metrics server URL.

        Returns
        -------

        str
            The metrics server URL.
        """
        return self.service_client.GET('metrics_server')

    def get_metrics_of_endpoints(self, metric, start_time, end_time, period):
        return self.service_client.GET('metrics/endpoints', {
            'metric':metric,
            'start_time':start_time,
            'end_time':end_time,
            'period':period})

    def get_metrics_of_endpoint(self, endpoint, metric, start_time, end_time, period):
        return self.service_client.GET('metrics/endpoints/'+endpoint, {
            'metric':metric,
            'start_time':start_time,
            'end_time':end_time,
            'period':period})

    def get_logs(self):
        """Returns the path that holds all of the log files.
        

        Returns
        -------

        str
        """
        return self.service_client.GET('logs')

    def get_log(self, log_name, start_time=None, end_time=None):
        """Returns the log paths to the logs corresponding to the time range.

        Parameters
        ----------

        log_name : str
            The name of the log. Should be one of:

            * server: The server logs.
            * graphlab_service: The evaluator logs.
            * query: The queries made.
            * result: Results of each query. 
            * feedback: The feedback.
            * audit: Everything that has been changed about the service.

        start_time : str
            The start time for the interval of logs you are interested in. If
            unspecified, it is 1 hour before the end time. Can be an ISO8601
            datetime string, UNIX epoch in seconds, or a number followed by
            "m", "h", or "d" indicating that many minutes, hours or day before
            now respectively.

        end_time : str
            The end time for the interval of logs you are interest in. If
            unspecfied, it is the current time. Can be an ISO8601 datetime
            string, UNIX epoch in seconds, or a number followed by "m", "h", or
            "d" indicating that many minutes, hours or days before now,
            respectively.

        Returns
        -------

        list of str
        """
        return self.service_client.GET('logs/'+log_name, {
            'start_time':start_time,
            'end_time':end_time})

    def flush_logs(self):
        return self.service_client.POST('logs/flush')

    def get_system_cache_configuration(self):
        """
        Returns the current system-wide cache configuration.

        Returns
        -------
        SystemCacheConfiguration
        """
        return SystemCacheConfiguration.from_json(self.service_client.GET('configurations/cache'))

    def set_system_cache_configuration(self, system_cache_configuration):
        """
        Updates the system-wide cache configuration.

        Parameters
        ----------
        system_cache_configuration : SystemCacheConfiguration
            The cache configuration parameters.

        Returns
        -------
        SystemCacheConfiguration
        """
        return SystemCacheConfiguration.from_json(
            self.service_client.PUT('configurations/cache',
                system_cache_configuration.to_json()))

    def get_endpoint_cache_configuration(self, endpoint_name):
        """
        Returns the current endpoint cache configuration for a single
        endpoint.

        Parameters
        ----------

        endpoint_name : str
            An endpoint name.

        Returns
        -------
        EndpointCacheConfiguration
            The cache configuration for that endpoint.
        """

        return EndpointCacheConfiguration.from_json(
            self.service_client.GET('configurations/cache/'+endpoint_name))

    def set_endpoint_cache_configuration(self, endpoint_name,
            endpoint_cache_configuration):
        """
        Returns the current endpoint cache configuration for a single
        endpoint.

        Parameters
        ----------

        endpoint_name : str
            An endpoint name.

        endpoint_cache_configuration : EndpointCacheConfiguration
            The cache configuration for that endpoint.

        Returns
        -------
        EndpointCacheConfiguration
            The cache configuration for that endpoint.
        """
        return EndpointCacheConfiguration.from_json(
            self.service_client.PUT('configurations/cache/'+endpoint_name,
                endpoint_cache_configuration.to_json()))

    def get_adhoc_execute_configuration(self):
        """Returns the adhoc execute configuration.

        Returns
        -------

        AdhocExecuteConfiguration

        See Also
        --------
        set_adhoc_executed_configuration
        """
        return AdhocExecuteConfiguration.from_json(
            self.service_client.GET('configurations/adhoc_execute'))

    def set_adhoc_execute_configuration(self, adhoc_execute_configuration):
        """Updates the adhoc execute configuration.

        Parameters
        ----------

        adhoc_execute_configuration : AdhocExecuteConfiguration

        Returns
        -------

        AdhocExecuteConfiguration

        See Also
        --------
        get_adhoc_executed_configuration
        """
        return AdhocExecuteConfiguration.from_json(
            self.service_client.PUT('configurations/adhoc_execute',
                adhoc_execute_configuration.to_json()))

    def clear_cache(self, name):
        """Clears the cache for Predictive Service."""
        return self.service_client.POST('configurations/cache/clear',
                        {'endpoint_name': name} if name else None)

    def restart_cache(self):
        """Restart the cache for Predictive Service."""
        return self.service_client.POST('configurations/cache/restart')

    def get_keys(self, type=None):
        """Returns all the keys of a specific type.

        Note that you must use an admin key to access this. All other keys
        will be denied access.

        Parameters
        ----------

        type : str
            The type of key desired, 'admin_key', 'api_key', or 'monitor_key'.

        Returns
        -------

        list of Key objects
        """
        return [
            Key.from_json(value)
            for value in self.service_client.GET('configurations/keys',{'type':type})
        ]

    def get_key(self, key_id='me'):
        """Returns the key given the key_id.

        Note that you must use an admin key to access other keys.

        Parameters
        ----------

        key_id : str
            The key_id of the key to fetch. If you use "me" it will return the
            key you are using.

        Returns
        -------

        Key object
        """
        return Key.from_json(self.service_client.GET('configurations/keys/'+key_id))


    def add_key(self, key):
        """Adds a new key to the system.

        Note that only admin keys can be used to create new keys.

        Parameters
        ----------

        key : Key, optional
            The Key object. If key_id is not set, one will be generated for
            you and returned in the response.

        Returns
        -------

        Key
            The Key object that was generated.

        """
        # The service will handle all the other errors.

        return Key.from_json(self.service_client.POST('configurations/keys',
            key.to_json()))


    def set_key(self, key):
        """Updates an existing key.

        Note that only admin keys can be used to create new keys.

        Note that the key_id or type cannot be modified. Thus, this is only
        useful to update an api_key endpoint_names attribute.

        Parameters
        ----------

        key : Key
            The Key object. The key_id must be set.

        Returns
        -------

        Key
            The Key object that was updated.

        """
        return Key.from_json(self.service_client.PUT('configurations/keys/'+key.id,
            key.to_json()))


    def remove_key(self, key_id=None, key=None):
        """Removes a key given its key_id.

        Parameters
        ----------

        key_id : str or Key, optional
            The Key or key id of the key to remove.

        key : Key, optional
            The Key to remove.

        Either key or key_id must be specified.
        """
        if not key_id:
            key_id = key

        if hasattr(key_id, 'id'):
            key_id = key_id.id
        return self.service_client.DELETE('configurations/keys/'+key_id)


    def get_environment_variables(self):
        """Returns the set of all environment variables.

        Returns
        -------

        dict

            Keys are the variable names, and values are the values.

        Example
        -------

            >>> mc.get_environment_variables()
            {"SQL_PASSWORD":"1234"}

        See Also
        --------

        get_environment_variables(), set_environment_variables(), remove_environment_variables(),
        add_environment_variable(), set_environment_variable(),
        remove_environment_variable()
        """
        return self.service_client.GET('configurations/env_vars')

    def set_environment_variables(self, env_vars):
        """Updates the entire set of all environment variables.

        Parameters
        ----------

        name : str
            The name of the environment variable.

        value : str
            The value of the environment variable.

        Returns
        -------

        str
            The value of the environment variable.

        See Also
        --------

        get_environment_variables(), set_environment_variables(), remove_environment_variables(),
        add_environment_variable(), set_environment_variable(),
        remove_environment_variable()
        """
        return self.service_client.PUT('configurations/env_vars', env_vars )

    def set_environment_variable(self, name, value):
        """Updates the entire set of all environment variables.

        Parameters
        ----------

        env_vars : dict
            The keys are the variable names, and the values are the variable
            values. Any missing variables will be deleted so that the
            environment variables match env_vars exactly.

        Returns
        -------

        dict

            Keys are the variable names, and values are the values.

        See Also
        --------

        get_environment_variables(), set_environment_variables(), remove_environment_variables(),
        add_environment_variable(), set_environment_variable(),
        remove_environment_variable()
        """
        return self.service_client.PUT('configurations/env_vars',
            {name:value})

    def remove_environment_variables(self):
        """Removes the entire set of environment variables.

        See Also
        --------

        get_environment_variables(), set_environment_variables(), remove_environment_variables(),
        add_environment_variable(), set_environment_variable(),
        remove_environment_variable()
        """
        return self.service_client.DELETE('configurations/env_vars')

    def add_environment_variable(self, name, value):
        """Adds a single environment variable.

        Parameters
        ----------

        name : str
            The name of the environment variable.

        value : str
            The value of the environment variable.

        Returns
        -------

        str
            The value of the environment variable.

        See Also
        --------

        get_environment_variables(), set_environment_variables(), remove_environment_variables(),
        add_environment_variable(), set_environment_variable(),
        remove_environment_variable()
        """
        return self.service_client.POST('configurations/env_vars',
            {name:value})

    def get_environment_variable(self, name):
        """Returns a single environment variable value.

        Parameters
        ----------

        name : str
            The name of the environment variable.

        Returns
        -------

        str
            The value of the environment variable.

        See Also
        --------

        get_environment_variables(), set_environment_variables(), remove_environment_variables(),
        add_environment_variable(), set_environment_variable(),
        remove_environment_variable()
        """
        return self.service_client.GET(
            'configurations/env_vars/' + name)

    def set_environment_variable(self, name, value):
        """Updates the value of a single environment variable.

        Parameters
        ----------

        name : str
            The name of the environment variable.

        value : str
            The value of the environment variable.

        Returns
        -------

        str
            The value of the environment variable.

        See Also
        --------

        get_environment_variables(), set_environment_variables(), remove_environment_variables(),
        add_environment_variable(), set_environment_variable(),
        remove_environment_variable()
        """
        return self.service_client.PUT(
            'configurations/env_vars/'+name,
            value)

    def remove_environment_variable(self, name):
        """Removes a single environment variable.

        Parameters
        ----------

        name : str
            The name of the environment variable.

        See Also
        --------

        get_environment_variables(), set_environment_variables(), remove_environment_variables(),
        add_environment_variable(), set_environment_variable(),
        remove_environment_variable()
        """
        return self.service_client.DELETE(
            'configurations/env_vars/'+name)

    def get_scale_factor_configuration(self):
        """Retrieves the scale factor configuration for the PredictiveService.

        Returns
        -------
        ScaleFactorConfiguration
        """
        return ScaleFactorConfiguration.from_json(
            self.service_client.GET('configurations/scale_factor'))

    def set_scale_factor_configuration(self, scale_factor):
        """Sets the scale factor configuration for the PredictiveService.

        Parameters
        ----------

        scale_factor : ScaleFactorConfiguration

        Returns
        -------

        ScaleFactorConfifuration
        """
        return self.service_client.POST('configurations/scale_factor',
            scale_factor.to_json())


    def get_cors_configuration(self):
        """Retrieves the CORS configuration for the PredictiveService.

        Returns
        -------
        CORSConfiguration
        """
        return CORSConfiguration.from_json(
            self.service_client.GET('configurations/CORS'))


    def set_cors_configuration(self, cors_configuration):
        """Sets the CORS configuration for the PredictiveService.

        Parameters
        ----------

        cors_configuration : CORSConfiguration

            The CORS configuration for the service.

        Returns
        -------

        CORSConfiguration
        """
        return self.service_client.PUT('configurations/CORS',
            cors_configuration.to_json())

    def consul_leave(self, host_ids):
        """
        Gracefully make the specified hosts to leave the consul cluster.

        Parameters
        ----------

        host_ids : list
            list of host_ids to leave the consul cluster.

        """
        return self.service_client.POST('consul_management/leave', {
            'host_ids':host_ids})

    def consul_repair(self):
        """
        Repair the consul cluster.

        This method is used to repair the consul cluster when the consul
        cluster goes in a bad state.
        """
        return self.service_client.POST('consul_management/repair', {
            'body':None})

    def get_consul_members(self):
        """
        Return member ship information regarding the consul

        Example return:

            Node                    Address             Status  Type    Build  Protocol  DC
            172.31.27.197           172.31.27.197:8301  alive   server  0.6.3  2         dc1
            172.31.27.197PS_SERVER  172.31.27.197:5301  alive   client  0.6.3  2         dc1
            172.31.27.198           172.31.27.198:8301  alive   server  0.6.3  2         dc1
            172.31.27.198PS_SERVER  172.31.27.198:5301  alive   client  0.6.3  2         dc1
        """
        return self.service_client.GET('consul_management/members')

    def get_consul_info(self):
        """
        Return current consul configuration

        Returns
        -------
        Output from "consul info" command, for example:
        redictive_service@6c89d82c9d23:/$ consul info
            agent:
                check_monitors = 0
                check_ttls = 0
                checks = 0
                services = 1
            build:
                prerelease =
                revision = c933efde
                version = 0.6.3
            consul:
                bootstrap = true
                known_datacenters = 1
                leader = true
                server = true
            raft:

        """
        return self.service_client.GET('consul_management/info')

    def get_consul_state(self):
        """
        Return current consul state

        Retrurns
        --------
        dictionary:

        For example:

            {u'master': u'172.31.31.27', u'state': u'good'}
        """
        return self.service_client.GET('consul_management/state')
