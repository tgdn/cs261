from re import compile as _compile
import requests

import time as _time
import types as _types
import glob as _glob
from datetime import datetime as _datetime
import json as _json

from collections import MutableMapping as _MutableMapping
from weakref import proxy as _weakref_proxy

from psclient.file_util import (
    is_path as _is_path,
    expand_full_path as _expand_full_path,
    mkdir as _mkdir,
    exists as _exists,
    remove_trailing_slash as _remove_trailing_slash,
    copy_from_local as _copy_from_local,
)

from .cluster_client import (
    predictive_service_environment_factory as
        _predictive_service_environment_factory,
    DockerPredictiveServiceEnvironment as _DockerPredictiveServiceEnvironment,
    LocalPredictiveServiceEnvironment as _LocalPredictiveServiceEnvironment
)

from .query_client import (
    QueryClient as _QueryClient,
)

from .rest import (
    RequestsNetworkWrapper as _RequestsNetworkWrapper,
    ServiceClient as _ServiceClient,
)

from .manager_client import (
    ManagerServiceClient as _ManagerServiceClient,
    Endpoint as _Endpoint,
    AliasEndpoint as _AliasEndpoint,
    Key as _Key,
    EndpointCacheConfiguration as _EndpointCacheConfiguration
)

from .config import generate_config as _generate_config

from .aws import get_credentials as _get_credentials

import os as _os

import logging as _logging
_logger = _logging.getLogger(__name__)
_name_checker = _compile('^[a-zA-Z0-9-_\ ]+$')

from .constants import (
    _NODE_LAUNCH_LIMIT,
    _DEFAULT_EVALUATOR,
    _R_MODEL_EVALUATOR_TYPE,
    _INTERNAL_PREDICTIVE_OBJECT_NAMES,
)


def _check_endpoint_name(name):
    """Checks that the endpoint name is valid by comparing it with an RE and
    checking that it is not reserved."""
    if not isinstance(name, basestring):
        raise TypeError("Endpoint name must be a string or unicode")

    if name == '':
        raise ValueError("Endpoint name cannot be empty")

    if not _name_checker.match(name):
        raise ValueError('endpoint name %r can only contain: a-z, A-Z, 0-9,'
            ' underscore, hyphens and spaces.' % name)

    if name in _INTERNAL_PREDICTIVE_OBJECT_NAMES:
        raise ValueError("Endpoint name \"%s\" is reserved" % name)

class PredictiveService(object):
    """
    Monitor/Manage a running Predictive Service.

    Predictive Service objects should not be instantiated directly, and are
    intended to be created using :func:`psclient.create`
    or loaded using :func:`psclient.connect`.
    """

    # State File Config Section Names
    _DEPLOYMENT_SECTION_NAME = 'Predictive Objects Service Versions'
    _PREDICTIVE_OBJECT_DOCSTRING = 'Predictive Objects Docstrings'
    _ENVIRONMENT_SECTION_NAME = 'Environment Info'
    _SERVICE_INFO_SECTION_NAME = 'Service Info'
    _SYSTEM_SECTION_NAME = 'System'
    _ENVIRONMENT_VARIABLE_SECTION_NAME = 'Environment Variables'
    _META_SECTION_NAME = 'Meta'

    # Directory Names
    _PREDICTIVE_OBJECT_DIR = 'predictive_objects'
    _DEPENDENCIES_DIR = 'dependencies'

    _typename = 'PredictiveService'

    def __init__(self,
            manager_endpoint=None,
            admin_key=None,
            api_key=None,
            query_endpoint=None,
            query_timeout=None,
            verify_certificate=True):
        """
        Parameters
        ----------

        manager_endpoint : str, optional

            The URL to the PS manager service. If not specified, the manager
            service functions are disabled.

        admin_key : str, optional

            An admin key. This will be the preferred key.

        api_key : str, optional

            An api key. This is used only if there is no admin_key specified.

        query_endpoint : str, optional

            The URL to the PS query service.

        query_timeout : float, optional

            The timeout for query operations.

        verify_certificate : bool, optional

            Whether to check the certificate for SSL connections. Defaults to
            True.

        """
        self._manager_endpoint = manager_endpoint
        self._admin_key = admin_key
        self._api_key = api_key
        self._query_endpoint = query_endpoint
        self._query_timeout = query_timeout
        self._verify_certificate = verify_certificate


    @property
    def _manager_service(self):
        """A client to the manager service. Generated from manager_endpoint"""
        if self._manager_endpoint is None:
            raise ValueError("Cannot connect to manager service. No"
                " manager_endpoint specified.")

        session = requests.session()
        session.verify = self._verify_certificate
        if self._admin_key is not None:
            session.auth = ('admin_key', self._admin_key)
        else:
            if self._api_key is not None:
                session.auth = ('api_key', self._api_key)
            else:
                raise ValueError("Cannot connect to manager service. No"
                    " admin_key or api_key specified.")

        # Setup the communications layer.
        network_wrapper = _RequestsNetworkWrapper(session)
        service_client = _ServiceClient(self._manager_endpoint, network_wrapper)

        return _ManagerServiceClient(service_client)

    @property
    def _query_service(self):
        """A client to the query service. Generated from query_endpoint."""
        if self._query_endpoint is None:
            self._query_endpoint = self.get_query_url()

        return _QueryClient(
            query_endpoint=self._query_endpoint,
            api_key=self._admin_key,
            query_timeout=self._query_timeout,
            verify_certificate=self._verify_certificate)

    @property
    def _info(self):
        """Deployment summary returned from the manager service. Cached for 1
        second."""
        return self._manager_service.get_service_info()

    @property
    def name(self):
        """The deployment name."""
        return self._info['name']

    @property
    def description(self):
        """The deployment description."""
        return self._info['description']

    @property
    def version(self):
        """The deployment version."""
        return self._info['server_version']

    @property
    def creation_time(self):
        """When the deployment was first created."""
        t = _datetime.fromtimestamp(int(self._info['creation_time']))
        return _datetime.isoformat(t)

    @property
    def _state_path(self):
        """The state file location. The state file contains critical
        information that all nodes refer to."""
        return self._info['state_path']

    @property
    def _environment(self):
        env_info = self._info['environment']
        if env_info.get('environment type') == 'Ec2PredictiveServiceEnvironment':
            # Get AWS credentials
            aws_access_key, aws_secret_key = _get_credentials()
            env_info['aws_credentials'] = {
                'aws_access_key_id': aws_access_key,
                'aws_secret_access_key': aws_secret_key
            }
        return _predictive_service_environment_factory(env_info)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        ret = ""
        ret += 'Name                  : %s' % self.name + '\n'
        ret += 'Version               : %s' % self.version + '\n'
        ret += 'Description           : %s' % self.description + '\n'
        ret += 'State Path            : %s' % self._state_path + '\n'
        ret += 'Creation time         : %s' % self.creation_time + '\n'
        ret += 'Query Endpoint        : %s' % self.get_query_url() + '\n'
        ret += 'Management Endpoint   : %s' % self._manager_endpoint + '\n'

        ret += "\nDeployed endpoints:\n"
        endpoint_strs = []
        endpoints = self.get_endpoints()
        for name in endpoints:
                endpoint_strs.append('\t%s [%s]\n' % (name, endpoints[name].type))
        endpoint_strs.sort()
        for s in endpoint_strs:
            ret += s

        return ret

    def get_query_url(self):
        '''
         Return the base url for querying the deployment's endpoints.

        Returns
        -------
        query_url : str
            The base url used for querying endpoints.
            To query and endpoint called "abc", use the following url:

                <base_url>/query/abc
        '''
        if not self._query_endpoint:
            self._query_endpoint = self._manager_service.get_query_url()

        return self._query_endpoint

    #
    # Environment Variables
    #
    class EnvironmentVariables(_MutableMapping):

        def __init__(self, ps):
            self._ps = _weakref_proxy(ps)

        def __repr__(self):
            return "{"+ \
                ", ".join([
                    repr(k) + ": " + repr(v)
                    for k, v in self.items()])+ \
            "}"

        def __getitem__(self, item):
            rv = self._ps._manager_service.get_environment_variable(item)
            if rv is None:
                raise KeyError(item)
            return rv

        def __setitem__(self, item, value):
            value = str(value)
            self._ps._manager_service.set_environment_variable(item, value)

        def __delitem__(self, item):
            self._ps._manager_service.remove_environment_variable(item)

        def _get_items(self):
            return self._ps._manager_service.get_environment_variables()

        def __len__(self):
            return len(self._get_items())

        def __iter__(self):
            return iter(self._get_items())

        def clear(self):
            """Clears all the environment variables"""
            self._ps._manager_service.remove_environment_variables()

        def update(self, *args, **kwargs):
            """Updates the environment variables in bulk."""
            env_vars = self._get_items()
            env_vars.update(*args, **kwargs)

            for k,v in env_vars.items():
                env_vars[k] = str(v)

            self._ps._manager_service.set_environment_variables(env_vars)

    @property
    def environ(self):
        """The environment variables in the Predictive Service.

        This is a dict-like object. Changing values in the dict, adding and
        removing keys, will change and add or remove environment variables on
        the Predictive Service. Note that the values will always be cast to
        str.

        If you set a dict-like object to environ, it will override the
        environment variables.

        If you ```del os.environ```, it will remove all the environment
        variables.
        """
        try:
            return self._environ
        except AttributeError:
            self._environ = self.EnvironmentVariables(self)
            return self._environ

    @environ.setter
    def environ(self, other):
        self._manager_service.set_environment_variables(other)

    @environ.deleter
    def environ(self):
        self._manager_service.remove_environment_variables()


    #
    # Keys
    #
    Key = _Key

    def get_keys(self, type=None):
        return self._manager_service.get_keys(type)
    get_keys.__doc__ = _ManagerServiceClient.get_keys.__doc__

    def get_key(self, key_id):
        return self._manager_service.get_key(key_id)
    get_key.__doc__ = _ManagerServiceClient.get_key.__doc__

    def add_key(self, key):
        return self._manager_service.add_key(key)
    add_key.__doc__ = _ManagerServiceClient.add_key.__doc__

    def set_key(self, key):
        return self._manager_service.set_key(key)
    set_key.__doc__ = _ManagerServiceClient.set_key.__doc__

    def remove_key(self, key_id=None, key=None):
        return self._manager_service.remove_key(key_id=key_id, key=key)
    remove_key.__doc__ = _ManagerServiceClient.remove_key.__doc__

    #
    # Status
    #

    def _get_nodes_status(self):
        '''
        Gets the status of each node in the environment.

        Examples
        --------

        >>> ps._get_nodes_status()
        '''
        node_status = {}
        for host in self.get_status().values():
            if not host.get('cache'):
                cache_state = 'Disabled'
            else:
                cache_state = 'Healthy' if host.get('cache',{}).get('healthy') else 'Unhealthy'

            node_status[host['id']] = {
                'dns_name':host['dns_name'],
                'ip_address':host['ip_address'],
                'private_ip_address':host['private_ip_address'],
                'state': host['state'],
                'reason': host['reason'],
                'cache': host['cache'],
            }

        return node_status


    def _get_endpoints_status(self):
        '''
        Get realtime endpoint status from  all nodes.

	    Returns
        -------
        dict

            Keys are endpoint names. Values are a dict with two keys.
            * ```'expected_version'``` is the highest version seen
            * ```'nodes'``` is a dictionary with keys of the node ids and
            values that are dicts. The keys of that dict are evaluator ids and
            the values are the evaluator status. The evaluator status is a
            dictionary with keys 'version', 'status', 'reason', 'type', and
            'cache_enabled'.


        Examples
        --------

        >>> ps._get_endpoints_status()

        {
           'ImagePredictor': {
                'expected_version': 2,
                'nodes':{
                    'i-29397b26':{
                        '9d01184030fa:1000': {
                            'version':1,
                            'status':'LoadSuccessful',
                            'reason':None,
                            'type':'model',
                            'cache_enabled':True,
                        },
                    },
                },
            },
        }
        '''
        from collections import defaultdict

        result = defaultdict( # Keys are endpoint names
            lambda: dict( # Keys are 'expected_version' and 'nodes'
                nodes=defaultdict( # Keys are node ids
                    defaultdict # Keys are evaluator ids
                )))

        for node_dns_name, node_status in \
                self._manager_service.get_status().items():

            node_id = node_status['id']

            # node_status['models'] is a list of dicts, but each dict has a
            # single entry with the key being the endpoint name. Let's join it
            # together.
            models = {}
            for model in node_status['models']:
                models.update(model)

            # models is a dict with keys being endpoint names and values being
            # a dict. That dict has keys of evaluator names and values of a
            # dict with keys 'cache_enabled', 'reason', 'status', and
            # 'version'.

            for endpoint_name, evaluator_statuses in models.items():
                for evaluator_id, eval_status in evaluator_statuses.items():
                    result[endpoint_name]['nodes'][node_id][evaluator_id] = \
                        eval_status

        # Calculate the expected_version by counting the max of the evaluator
        # versions.
        for endpoint_name, endpoint_status in result.items():
            expected_version = 0
            type = None
            for node_id, node_status in endpoint_status['nodes'].items():
                for evaluator_id, evaluator_status in node_status.items():
                    expected_version = max(
                        expected_version,
                        evaluator_status['version'])
                    type = type or evaluator_status['type']
            endpoint_status['expected_version'] = expected_version
            endpoint_status['type'] = type

        # Convert to dicts
        for endpoint_name, endpoint_status in result.items():
            for node_id, node_status in endpoint_status['nodes'].items():
                endpoint_status['nodes'][node_id] = dict(node_status)
            endpoint_status['nodes'] = dict(endpoint_status['nodes'])
        result = dict(result)


        return result


    def _get_cache_status(self):
        '''
        Get the cache status for all deployed Predictive Objects on each node.

        Examples
        --------

        >>> ps._get_cache_status()
            {
                'endpoint name': {
                    'node id': {
                        'evaluator id': True,
                    },
                },
            }

        '''
        result = self._get_endpoints_status()

        for endpoint_name, endpoint_status in result.items():
            result[endpoint_name] = endpoint_status['nodes']

        for endpoint_name, nodes in result.items():
            for node_id, evaluators in nodes.items():
                for evaluator_id, evaluator_status in evaluators.items():
                    evaluators[evaluator_id] = \
                        evaluator_status['cache_enabled']

        return result


    def get_status(self, view=None):
        '''
        Gets the status of the current Predictive Service.

        If parameter 'view' is given, will return an SFrame that shows specific
        aspect of the status.

        Parameters
        ----------
        view : 'cache'|'node'|'endpoint'

        Returns
        -------
        out : list | SFrame
            If `view` is not given, then return a list with each element being
            a dictionary containing status information for one node. Available
            keys are: ['dns_name', 'id', 'models', 'state']

            If 'view' is 'cache', returns an SFrame showing cache status for each
            Predictive Object deployed in each node. Columns in SFrame include
            ['name', 'node.instance-id1', 'node.instance-id2', etc.], see sample
            output below. If no Predictive Object deployed yet, then returns None.

            If 'view' is 'node', returns an SFrame showing node status for each
            node in current Predictive Service. Columns in SFrame include
            ['dns_name', 'instance-id', 'state']

            If 'view' is 'endpoint', returns an SFrame showing status for each
            deployed endpoint in each node. Columns in SFrame include
            ['name', 'expected version', 'type', 'node.instance-id1', 'node.instance-id2',
            etc...], see sample output below. If no Predictive Object deployed yet,
            then returns None.

        Examples
        --------

        This is a sample output for a Predictive Service with two nodes and two
        Predictive Objects deployed. With one cache enabled and one disabled.
        (Note: output omitted some SFrame header information.)

        >>> ps.get_status()
            [{'dns_name': 'ec2-54-148-131...',
              'id': 'i-2a397b25',
              'state': 'InService',
              'models': [
                {
                    u'cache_enabled': True,
                    u'description': u'',
                    u'name': u'Image Predictor',
                    u'status': u'Loaded successfully',
                    u'version': 1
                }, {
                    u'cache_enabled': False,
                    u'description': u'',
                    u'name': u'Book Recommender',
                    u'status': u'Loaded successfully',
                    u'version': 2
               }],
              },
             {'dns_name': 'ec2-54-186-64-...',
              'id': 'i-29397b26',
              'state': 'InService',
              'models': [
                {
                    u'cache_enabled': True,
                    u'description': u'',
                    u'name': u'Image Predictor',
                    u'status': u'Loaded successfully',
                    u'version': 1
                }, {
                    u'cache_enabled': False,
                    u'description': u'',
                    u'name': u'Book Recommender',
                    u'status': u'Loaded successfully',
                    u'version': 2
               }],
              },
            ]

        >>> ps.get_status(view='cache')
        +-----------------+-------------------------+-------------------------+
        |       name      |     node.i-29397b26     |     node.i-2a397b25     |
        +-----------------+-------------------------+-------------------------+
        | Image Predictor |       1 (Enabled)       |      1 (Enabled)        |
        | Book Recommender|       0 (Disabled)      |      0 (Disabled)       |
        +-----------------+-------------------------+-------------------------+

        >>> ps.get_status(view='node')
        +--------------------+---------------+-----------+-----------+
        |      dns_name      |  instance_id  |   state   |   cache   |
        +--------------------+---------------+-----------+-----------+
        | ec2-54-148-131 ... |  i-2a397b25   | InService | Healthy   |
        | ec2-54-186-64- ... |  i-29397b26   | InService | Healthy   |
        +--------------------+---------------+-----------+-----------+

        >>> ps.get_status(view='endpoint')
        +-----------------+------------------+--------+-------------------------+-------------------------+
        |       name      | expected version |  type  |     node.i-29397b26     |     node.i-2a397b25     |
        +-----------------+------------------+--------+-------------------------+-------------------------+
        | Image Predictor |        1         | model  | 1 (Loaded successfully) | 1 (Loaded successfully) |
        | Book Recommender|        2         | policy | 2 (Loaded successfully) | 2 (Loaded successfully) |
        +-----------------+------------------+--------+-------------------------+-------------------------+

        '''
        if not view:
            return self._manager_service.get_status()

        if view == 'cache':
            return self._get_cache_status()

        elif view == 'node':
            return self._get_nodes_status()

        elif view == 'endpoint':
            return self._get_endpoints_status()
        else:
            raise ValueError("Supported views are: ['cache', 'node',"
                " 'endpoint']; '%s' is not supported." % view)

    #
    # Query
    #

    @property
    def query_timeout(self):
        """The timeout for queries in seconds."""
        return self._query_service.query_timeout

    @query_timeout.setter
    def query_timeout(self, value):
        self._query_service.query_timeout = value

    def query(self, name, *args, **kwargs):
        return self._query_service.query(name, *args, **kwargs)
    query.__doc__ = _QueryClient.query.__doc__


    def feedback(self, key, **kwargs):
        return self._query_service.feedback(key, **kwargs)
    feedback.__doc__ = _QueryClient.feedback.__doc__

    #
    # Endpoints
    #

    def get_endpoints(self, type=None):
        """Returns the endpoints.

        See ManagerServiceClient.get_endpoints for a full description.
        """
        return self._manager_service.get_endpoints(type)

    def _get_endpoint_upload_destination(self):
        """Returns the endpoint upload destination."""
        return self._manager_service.get_endpoint_upload_destination()['path']

    def alias(self, alias, existing_endpoint_name, description = None):
        '''
        Create a new endpoint to redirect to an existing endpoint, or update an
        existing alias to point to a different existing endpoint.

        You cannot update an exiting endpoint that is not already an alias.

        Parameters
        ----------
        alias : str
            The new endpoint name or an existing alias endpoint name.

        existing_endpoint_name : str
            A name of an existing endpoint to redirect the alias to.

        description : str, optional
            A description for the alias.
        '''
        # check for invalid PO names
        _check_endpoint_name(alias)

        if not description:
            description = 'Alias for %s' % existing_endpoint_name

        if existing_endpoint_name not in self.get_endpoints():
            raise ValueError("Endpoint '%s' does not exist." % existing_endpoint_name)

        # Can only overwrite existing alias
        existing_endpoint = self.get_endpoints().get(alias)
        endpoint = _AliasEndpoint(
                name          = alias,
                type          = 'alias',
                description   = description,
                target        = existing_endpoint_name,
                cache_state   = 'disabled',
                version       = 1,
                schema_version= 1)

        if existing_endpoint:
            if existing_endpoint.type != 'alias':
                raise RuntimeError('Name "%s" is already in use by another endpoint.' % alias)

            endpoint.version = existing_endpoint.version + 1

            self._manager_service.set_endpoint(endpoint)
        else:
            self._manager_service.add_endpoint(endpoint)

        self._wait_for_endpoint_deployment(alias, endpoint.version)

    def deploy(self,
        name, obj, description='', evaluator=_DEFAULT_EVALUATOR,
        schema=None,
        override=False):
        """Deploys an endpoint.

        The endpoint may be backed by one of the following:

           * a GraphLab Create model
           * a user-defined function, or
           * a group of models tied together by an `EndpointPolicy`

        This operation will not take effect until `apply_changes` is called.

        Parameters
        ----------
        name : str
            A unique identifier for the endpoint. This name is used
            when querying the Predictive Service.

        obj :  `graphlab.Model` | str | func | `EndpointPolicy`
            The actual object that backs the endpoint.

            If 'obj' is a str, it is interpreted as a path (local/S3/HDFS) to where
            a GraphLab Create Model is persisted.

            If 'obj' is a user-defined function, the query can have any signature.
            However both input and output of the query need to be JSON serializable

            If 'obj' is an EndpointPolicy, it has to be one of the following
            supported policies:

                * ProbabilityPolicy
                * EpsilonGreedyPolicy

        description : str, optional
            The description for the endpoint. The __doc__ string of the object
            will be used if this is None.

        evaluator : str, optional
            The specified evaluator for this endpoint. Supported evaluators
            are: [`python27`,`R`].  Default evaluator is `python27`.

        schema : dict, optional
            The schema of the obj (only valid for user-defined function).
            The schema contains information such as the input parameters, sample input,
            and the output parameters.
            It is advisable to have a schema for custom predictive object so that
            future users of the Predictive Service know how to query.
            Refer to graphlab.deploy.util.generate_schema for more information on
            how to generate the schema.

        override : bool
            Whether to override (update) an existing endpoint. If False and
            there is already an endpoint with that name, it will raise a
            RuntimeError. If True and there is already an endpoint with that
            name, it will deploy a new version on top of it.

        See Also
        --------
        remove, get_endpoints

        Examples
        --------
        To add a GraphLab Create Model:

            >>> model = graphlab.recommender.create(...)
            >>> ps.deploy('recommender', model)

        To add a GraphLab Create Model saved locally / S3:

            >>> ps.deploy('recommender', '~/saved_models/recommender')
            >>> ps.deploy('recommender2', 's3://model-archive/recommender')

        To add an endpoint backed by a user-defined function:

            >>> def add(a, b):
            >>>     return a + b
            >>>
            >>> ps.deploy('add_two_numbers', add)
            >>> ps.apply_changes()
            >>> ps.query('add_two_numbers', a = 1, b = 2)

        To add a user-defined function that has GraphLab Create model dependency:

            >>> recommender = graphlab.recommender.create(...)
            >>> def my_recommender(user_id):
            >>>
            >>>     if not isinstance(user_id, str):
            >>>         raise ValueError('"user_id" has to be a string')
            >>>     return recommender.recommend([user_id])
            >>>
            >>> ps.deploy('custom-recommender', my_recommender)

            >>> ps.apply_changes()
            >>> ps.query('custom-recommender', user_id = 'abc')

        If the custom query depends on other Python package(s), the dependendent
        package(s) may be declared using @required_packages decorator:

            >>> from psclient.decorators import required_packages
            >>> @required_packages(["names", ...])
            >>> def generate_names(num_items):
            >>>     import names
            >>>     return gl.SArray([names.get_full_name() for i in range(num_items)])

        To add an endpoint backed by `EndpointPolicy`:

            >>> from psclient.policy import ProbabilityPolicy
            >>> p = ProbabilityPolicy({'m1':0.8, 'm2':0.2})
            >>> ps.deploy('endpoint-name', p)

        To add an endpoint backed by a schema:
            >>> schema = graphlab.deploy.util.generate_schema(
                            input = {'x' : 3, 'y': 2},
                            output = 6,
                            input_description = {'x': 'value of x',
                                                 'y': 'value of y'},
                            output_description = 'the value of multiplying')
            >>> ps.add('multiplying', mutiplying, 'a mathematical operation', schema))
        """
        endpoint = self.get_endpoints().get(name)
        if endpoint:
            if not override:
                raise RuntimeError("An endpoint with that name (%r) already"
                    " exists. Use 'override = True' to force update "
                    "an existing endpoint." % name)

            version = endpoint.version + 1
        else:
            version = 1

        obj = self._gen_endpoint(name, obj, description, evaluator, version,
            schema)

        self._upload_endpoint(obj)

        if version == 1:
            self._manager_service.add_endpoint(_Endpoint(**obj))
        else:
            self._manager_service.set_endpoint(_Endpoint(**obj))

        self._wait_for_endpoint_deployment(obj['name'], obj['version'])

    def _gen_endpoint(self, name, obj, description, evaluator, version=1,
            schema=None):
        '''Generates an endpoint dict.

        Parameters
        ----------
        name : str
            Endpoint name to add or update

        obj :  | `graphlab.Model` | str | func | `EndpointPolicy`
            Object that backs the endpoint. See add() for a complete
            description.

        description : str
            Description of the endpoint

        evaluator: str
            The evaluator that corresponds to the obj

        version : int
            The version. Defaults to 1.

        Returns
        -------

        dict

            Keys:

                name : str
                    The name provided.

                version : int
                    The version provided.

                schema_version : int
                    The schema version of this endpoint.

                description : str
                    The provided description.

                type : str
                    The type of the endpoint. Can be 'model' or 'policy'.
                    ('alias' is set by a different method.)

                endpoint_obj : object
                    The wrapper around the obj provided that can be used to
                    generate the code and dependencies for the endpoint.

        Raises
        ------

        TypeError
            When obj is not one of the expected types.
        '''
        # check for invalid PO names
        _check_endpoint_name(name)

        if description is None:
            if isinstance(obj.__doc__, basestring):
                # extract doc string
                description = obj.__doc__.strip() or ''
            else:
                description = ''

        endpoint_type, endpoint_object, methods = self._create_endpoint_object(obj,
            description, evaluator, schema)

        return {
            'name'          : name,
            'version'       : version,
            'schema_version': endpoint_object.schema_version,
            'description'   : description,
            'type'          : endpoint_type,
            'endpoint_obj'  : endpoint_object,
            'dependencies'  : endpoint_object.get_dependencies(),
            'evaluator'     : evaluator,
            'methods'       : methods,
            'required_files': [],
            'required_packages': [],
        }

    def _create_endpoint_object(self, obj, description, evaluator, schema):
        from psclient.policy import EndpointPolicy
        from psclient.predictive_object import PredictiveObject
        from psclient.custom_query_predictive_object import CustomQueryPredictiveObject
        from psclient.r_predictive_object import RPredictiveObject

        """Returns (type, object) for the obj passed in."""
        if isinstance(obj, _types.FunctionType):
            obj = CustomQueryPredictiveObject(
                query=obj,
                description=description,
                schema=schema)
            return ('model', obj, obj.get_methods())

        if isinstance(obj, str) and evaluator == _R_MODEL_EVALUATOR_TYPE:
            obj = RPredictiveObject(obj)
            return ('model', obj, obj.get_methods())

        if isinstance(obj, PredictiveObject):
            return ('model', obj, obj.get_methods())

        if isinstance(obj, EndpointPolicy):
            methods = self._validate_policy_endpoint(obj)
            return ('policy', obj, methods)

        # We can't load graphlab or sframe but we do need to see if it is
        # already loaded.
        import sys
        _gl = sys.modules.get('graphlab')
        if _gl:
            from psclient.model_predictive_object import ModelPredictiveObject
            if _is_path(obj) or \
                    isinstance(obj, _gl.Model) or \
                    isinstance(obj, _gl.CustomModel):
                obj = ModelPredictiveObject(model=obj)
                return ('model', obj, obj.get_methods())

        raise TypeError("'obj' parameter can only be one of the following: "
                        "an instance of GraphLab Create Model, "
                        "a path to a saved GraphLab Create Model, "
                        "an user-defined function, "
                        "or an instance of EndpointPolicy. "
                        "If you are providing a GraphLab Create Model, you "
                        "need to have GraphLab Create installed.")

    def _validate_policy_endpoint(self, policy):
        '''
        Validate the given policy. All models used in the policy
        should have been deployed

        Parameters
        -----------
        policy : EndpointPolicy
            The policy to be added to current Predictive Service
        '''
        from psclient.policy import EndpointPolicy
        if not isinstance(policy, EndpointPolicy):
            raise TypeError('Policy must be a type of psclient.policy.EndpointPolicy')

        # We require that all models involved in the policy have been deployed.
        policy_models = policy.get_models()

        endpoints = self.get_endpoints()
        for m in policy_models:
            if m not in endpoints:
                raise RuntimeError('Model "%s" used in policy has not been deployed.' \
                ' Use get_endpoints to get all deployed models.' \
                    % (m))

        endpoints_in_policy = []
        for m in policy_models:
            if m in endpoints:
                endpoints_in_policy.append(endpoints[m])
        self._verify_schema_match(endpoints_in_policy)
        methods = endpoints_in_policy[0].methods

        return methods


    def _verify_schema_match(self, endpoints_in_policy):
        """Verify that the endpoints in the policy all match with the same
        schema."""

        # Check the number of methods is all the same
        if len(set([len(p.methods) for p in endpoints_in_policy])) != 1:
            raise TypeError("Unequal number of methods for models in policy.")

        for i in range(1, len(endpoints_in_policy)):
            # The schema might be None, which is okay if all the endpoints
            # do not have schema.
            if endpoints_in_policy[i].methods[0]['schema'] != \
                endpoints_in_policy[0].methods[0]['schema']:
                # The input and sample of schemas of the endponts should be the same,
                # but the output might not be the same.
                if endpoints_in_policy[i].methods[0]['schema']['input'] \
                    != endpoints_in_policy[0].methods[0]['schema']['input']:
                    raise TypeError('Not all endpoints have the same schema input.')
                if endpoints_in_policy[i].methods[0]['schema']['sample'] \
                    != endpoints_in_policy[0].methods[0]['schema']['sample']:
                    raise TypeError('Not all endpoints have the same schema sample.')

    def _upload_endpoint(self, obj):
        """Sends the endpoint across the wire."""
        endpoint_obj = obj['endpoint_obj']

        dest_path = self._get_endpoint_upload_destination()

        # Upload the endpoint
        obj['src_path'] = _os.path.join(
            dest_path,
            'endpoints',
            obj['name'],
            str(obj['version']))
        endpoint_obj.save(obj['src_path'])

        # Upload required files for the Predictive Object
        if hasattr(endpoint_obj, 'required_files') and \
                endpoint_obj.required_files:

            obj['required_files'] = _os.path.join(
                dest_path,
                'required_files',
                obj['name'],
                str(obj['version']))

            for remote_path, local_path in endpoint_obj.required_files.iteritems():
                self._upload_file(
                    dest_path=obj['required_files'],
                    local_path=local_path,
                    remote_path=remote_path)

        # upload dependent package files for Predictive Object
        if hasattr(endpoint_obj, 'required_packages') and \
                endpoint_obj.required_packages:

            obj['required_packages_dst_path'] = _os.path.join(
                dest_path,
                'required_packages',
                obj['name'],
                str(obj['version']))

            if not _exists(obj['required_packages_dst_path']):
                _mkdir(obj['required_packages_dst_path'])

            obj['required_packages'] = []
            for package_name_or_file in endpoint_obj.required_packages:
                local_path = _expand_full_path(package_name_or_file)
                package_name = _os.path.basename(local_path)
                self._upload_file(
                    dest_path = obj['required_packages_dst_path'],
                    local_path = local_path,
                    remote_path = package_name)

                obj['required_packages'].append(package_name)


    def _upload_file(self, dest_path, local_path, remote_path = ''):
        '''
        Add files to Predictive Service, this could be any file type. The files are
        put into a fixed folder (DEPENDENT-FILES-ROOT) in Predictive Service
        nodes and are organized into directory structure.

        If the file or folder exists already, the files/folders are overwritten.

        Parameters
        -----------
        dest_path : str
            A file URL where the required files will be loaded into. IE,
            "s3://...".

        local_path : str
            A local file path, could be a directory or a file. "glob" style files
            name matching is supported.

        remote_path : str, optional
            Remote file path. This has to be a relative file path. The path will
            be relative to the remote (DEPENDENT-FILES-ROOT) directly.
            If not given, the remote path will get the name of either file or
            directory in local_path
        '''

        # turn to absolute path with no trailing slash
        local_path = _expand_full_path(local_path)
        local_path = _remove_trailing_slash(local_path)

        if remote_path:
            if _os.path.isabs(remote_path):
                raise ValueError('remote_path has to be a relative path, it cannot starts with "/".')

            # Do not allow going up the file path
            if '..' in remote_path:
                raise ValueError('".." is not allowed to be in remote path.')

        files = _glob.glob(local_path)

        for f in files:
            if remote_path:
                remote_file_path = dest_path + '/' + remote_path
            else:
                remote_file_path = dest_path + '/' + _os.path.basename(f)

            _copy_from_local(f, remote_file_path)

    def _wait_for_endpoint_deployment(self,
            endpoint_name,
            version=1,
            interval=1.0,
        ):
        """
        Waits for the endpoint to be deployed by calling get_status() and
        checking the versions deployed of the endpoint against the expected
        version. If all the versions are equal to or greater than the version
        expected, then it will return. Uses time.sleep().
        """
        _logger.info("Waiting for endpoint %r to deploy to version %r",
            endpoint_name,
            version)
        while True:
            ep_status = self.get_status('endpoint')
            try:
                ep = ep_status[endpoint_name]
            except KeyError:
                _logger.info("Endpoint %r doesn't exist in endpoints yet",
                    endpoint_name)
            else:
                _logger.info("ep=%r", ep)

                statuses = [
                    evaluator_status['status']
                    for node_status in ep['nodes'].values()
                    for evaluator_status in node_status.values()
                ]

                if 'LoadFailed' in statuses:
                    raise Exception("LoadFailed", ep)

                success = [
                    evaluator_status['version'] >= version
                        and evaluator_status['status'] == 'LoadSuccessful'
                    for node_status in ep['nodes'].values()
                    for evaluator_status in node_status.values()
                ]

                if all(success):
                    _logger.info("Deployment is successful.")
                    return
                else:
                    _logger.info("%s of %s evaluators are completed.",
                        success.count(True), len(success))

            _logger.info("Sleeping %r", interval)
            _time.sleep(interval)

    def remove(self, name):
        '''
        Remove the alias, custom method, or model that has the specified name.

        Parameters
        ----------
        name : str
            The name of the endpoint to be removed.

        Notes
        -----
        This could fail if the endpoint does not exist, or if the endpoint is
        in use by other endpoints like alias or policy. To check all endpoints
        that are depending on this endpoint, use `get_endpoint_dependencies`.

        See Also
        --------
        deploy, get_endpoint_dependencies
        '''
        self._manager_service.remove_endpoint(name)

        # Wait for the endpoint to be removed
        while name in self.get_endpoints():
            _time.sleep(1.0)

    def get_endpoint_dependencies(self, endpoint_name = None):
        '''
        Get all endpoints that depend on the given endpoint.

        Parameters
        ----------
        endpoint_name : str, optional
            The name of the endpoint to find dependendent endpoints. If not given,
            find all dependendent endpoints for all endpoints

        Returns
        -------
        dependendent endpoints : list of str or dict of str -> list of str
            if endpoint_name is given, returns a list of endpoint names that depend
            on the given endpoint.

            If endpoint_name is not given, returns a dictionary where key is the
            endpoint name and value is a list of endpoints that depend on the
            endpoint specified by the key.

        '''
        endpoints = self.get_endpoints()

        def get_dependencies(endpoint):
            result = set()
            for d in endpoints[endpoint].dependencies:
                result.update([d])
                result.update(get_dependencies(d))
            return result

        if endpoint_name:
            return get_dependencies(endpoint_name)

        else:
            return {
                endpoint : get_dependencies(endpoint)
                for endpoint in endpoints
            }

    def get_config_params(self):
        """Returns the parameters used to instantiate this object.

        Returns
        -------

        dict

            Keys are:

            * manager_endpoint
            * admin_key
            * api_key
            * query_endpoint
            * query_timeout
            * verify_certificate

            Values are the values specified when the object was created.
        """
        return {
            k:getattr(self, '_'+k)
            for k in (
                'manager_endpoint',
                'admin_key',
                'api_key',
                'query_endpoint',
                'query_timeout',
                'verify_certificate',
            )
        }

    def generate_config(self,
            config_file,
            **kwargs):
        """Generates a config file for the current PredictiveService instance.

        The keys of the **kwargs dict are configuration parameters.

        * If the key is missing, it will be loaded from this object.

        * If the value are specified, these will be used, overriding whatever
          is in this object.

        * If the value is None, then the key will not be used.

        The keys are:

        * manager_endpoint
        * query_endpoint
        * admin_key
        * api_key
        * verify_certificate
        * query_timeout

        Parameters
        ----------

        config_file : str or file
            The file the config will be written to.

        **kwargs : dict
            Parameters you'd like to override or not use.

        Examples
        --------

        If you'd like to create a config file that copies this object exactly:

        >>> ps.generate_config('test.conf')

        If you'd like to create a config file that contains an api_key and
        this object's configuration except for the manager_endpoint and
        admin_key:

        >>> ps.generate_config('test.conf',
                manager_endpoint=None,
                admin_key=None,
                api_key='your api key')
        """

        _generate_config(
            config_file,
            self,
            **kwargs)


    #
    # Cache
    #
    def get_cache_state(self, name=None):
        '''
        Return current cache configurations. If name is None, returns cache
        state for all endpoints. If specified, returns only the cache state
        for a given endpoint

        Parameters
        ----------
        name : str, optional
            Endpoint name

        Returns
        ------

        str or dict

        Example
        -------

        Calling with no parameters will return the global_cache_state
        configuration and the cache state for each endpoint.

        >>> ps.get_cache_state()
        {'global_cache_state': 'enabled', 'endpoints': {u'add': 'enabled'}}


        Calling with an endpoint name will return 'enabled' or 'disabled' for
        that endpoint.

        >>> ps.get_cache_state('add')
        'enabled'

        '''
        if name:
            return self._manager_service.get_endpoint_cache_configuration(name).cache_state
        else:
            global_cache_state = self.get_configuration('global_cache_state')['global_cache_state']
            endpoints = {}
            for endpoint in self.get_endpoints():
                endpoints[endpoint] = self._manager_service.get_endpoint_cache_configuration(endpoint).cache_state
            return {
                'global_cache_state': global_cache_state,
                'endpoints': endpoints}

    def cache_enable(self, name, restart=True):
        """Enable caching for an endpoint.

        In addition to calling cache_enable for the endpoints you would like
        caching enabled for, you must also turn on system-wide caching with
        configure().

        Parameters
        ----------
        name : str
            The endpoint name.

        restart : bool, optional (defaults to True)
            Whether to flush and restart the entire cache.
        """
        self._manager_service.set_endpoint_cache_configuration(
            name,
            _EndpointCacheConfiguration(cache_state='enabled'))

        if restart:
            self._manager_service.restart_cache()

    def cache_disable(self, name):
        """Disable caching for an endpoint.

        Parameters
        ----------
        name : str
            The endoint name.
        """
        self._manager_service.set_endpoint_cache_configuration(
            name,
            _EndpointCacheConfiguration(cache_state='disabled'))

    def cache_clear(self, name=None):
        """Clears the cache for a single endpoint or the entire service.

        Parameters
        ----------
        name : str, optional
            The endpoint name. If not specified, the entire cache is cleared.
        """
        self._manager_service.clear_cache(name)

    #
    # Node manipulations.
    #
    def _remove_nodes(self, instance_ids):
        '''
        Terminate one or more nodes in a Predictive Service cluster by their
        instance IDs, and remove them from the cluster.

        Parameters
        ----------

        instance_ids : list[str]
            A list of instance ids to remove
        '''
        if not isinstance(instance_ids, list):
            instance_ids = [instance_ids]

        if not all([type(i) in [str, unicode] for i in instance_ids]):
            raise TypeError('The "instance_ids" parameter must be either a ' \
                            'string or a list of strings')

        instance_ids = list(set(instance_ids)) # ensure unique elements

        node_ids = {x.id for x in self._environment._get_all_hosts()}
        num_nodes_deployed = len(node_ids)

        # ensure we're not removing any nodes outside this predictive service
        if not all([instance_id in node_ids for instance_id in instance_ids]):
            raise ValueError("Cannot remove a node that is not in this " \
                             "PredictiveService cluster.")

        if num_nodes_deployed == len(instance_ids):
            raise RuntimeError("Terminating all nodes in the cluster is not supported.")

        # leave consul cluster
        self._manager_service.consul_leave(instance_ids)

        self._environment.terminate_instances(instance_ids)

    def remove_nodes(self, instance_ids):
        '''
        Terminate one or more nodes in a Predictive Service cluster by their
        instance IDs, and remove them from the cluster.

        This is useful in the event of a node failure. You can query the
        Predictive Service status with `get_status`. If there are unresponsive
        nodes, capture their instance IDs and pass them as the sole parameter
        to this function.

        Parameters
        ----------
        instance_ids : list[str] | str
            The instance IDs for the nodes to terminate.

        See Also
        --------
        get_status, add_nodes, cache_enable

        Notes
        -----
        Calling this method will clear out any existing data in the distributed
        cache.

        Examples
        --------
        To remove a node in the current Predictive Service:

          >>> print ps.get_status('node')
          +---------+-------------------------------+-------------+-----------+
          |  cache  |            dns_name           | instance_id |   state   |
          +---------+-------------------------------+-------------+-----------+
          | Healthy | ec2-12-11-12-121.us-west-2... |  i-4399eb8a | InService |
          | Healthy | ec2-21-21-24-221.us-west-2... |  i-ccc4663a | InService |
          +---------+-------------------------------+-------------+-----------+
          [2 rows x 4 columns]
          >>> ps.remove_nodes(['i-ccc4663a'])
          >>> print ps.get_status('node')
          +---------+-------------------------------+-------------+-----------+
          |  cache  |            dns_name           | instance_id |   state   |
          +---------+-------------------------------+-------------+-----------+
          | Healthy | ec2-12-11-12-121.us-west-2... |  i-4399eb8a | InService |
          +---------+-------------------------------+-------------+-----------+
          [1 rows x 4 columns]

        '''
        self._raise_error_if_on_premise_ps()
        self._remove_nodes(instance_ids)

    def _add_nodes(self, num_nodes, instance_type=None,
                   security_group_name=None, CIDR_rule=None, tags=None):
        '''
        Add one or more nodes to a Predictive Service Cluster.

        Parameters
        ----------
        num_nodes : int
            The number of additional nodes to add to the cluster. The maximum
            number of nodes that can be launched at one time is 5. If the
            `num_nodes` parameter exceeds that limit, a ValueError is thrown.

        instance_type : str, optional
            The type of instance to launch. If specified, this must match the
            instance type used when the service was originally launched.

        security_group : str, optional
            The name of the security group for the EC2 instance to use. If
            specified, this must match the instance type used when the service
            was originally launched.

        CIDR_rule : string or list[string], optional
            The Classless Inter-Domain Routing rule(s) to use for the instance.
            Useful for restricting the IP Address Range for a client. Default is
            no restriction. If you specify CIDR_rule(s), you must also specify a
            security group to use.

        tags : dict, optional
            A dictionary containing the name/value tag pairs to be assigned to
            the instance. If you want to create only a tag name, the value for
            that tag should be the empty string (i.e. ''). In addition to these
            specified tags, a 'GraphLab' tag will also be assigned.
        '''

        if num_nodes <= 0:
            _logger.warn("The num_nodes parameter must have a positive " \
                         "value for this method to have any effect.")
            return

        if num_nodes > _NODE_LAUNCH_LIMIT:
            raise ValueError("You cannot launch more than %d nodes. If this "
                             "limit is problematic, please visit "
                             "https://turi.com/support for support options." %
                             _NODE_LAUNCH_LIMIT)

        if not instance_type:
            attrs = self._environment._get_instance_attributes()
            instance_type = attrs['instance_type']
            tags = attrs['tags']
            security_group_id = attrs['security_group_id']
            subnet_id = attrs['subnet_id']

        additional_ports_to_open = [self._environment.admin_port,
                                    self._environment.admin_ui_port]

        self._environment.add_instances(self._state_path, num_nodes,
                                        instance_type,
                                        tags,
                                        security_group_id = security_group_id,
                                        subnet_id = subnet_id,
                                        CIDR_rule = CIDR_rule,
                                        additional_ports_to_open = additional_ports_to_open,
                                        existing_node_private_ip = attrs['private_ip_address'])

    def list_custom_packages(self):
        '''
        List all custom Python packages (eggs) that have been uploaded to the
        predictive service. The mechanism to upload such packages is the
        :py:meth:`required_packages()` decorator.

        Returns
        -------

        packages : list(str)
            A list of custom Python packages (eggs) that have been uploaded.

        Examples
        --------

            >>> ps.list_custom_packages()
            out:
                ['my-custom-package.tar.gz', 'another-package.tar.gz']

        '''
        return self._manager_service.get_custom_packages()

    def add_nodes(self, num_nodes):
        '''
        Add one or more nodes to a Predictive Service Cluster.

        Parameters
        ----------
        num_nodes : int
            The number of additional nodes to add to the cluster. The maximum
            number of nodes that can be launched at one time is 5. If the
            `num_nodes` parameter exceeds that limit, a ValueError is thrown.

        Notes
        -----
        Calling this method will clear out any existing data in the distributed
        cache.

        Examples
        --------
        To add three new nodes to the current Predictive Service:

          >>> ps.add_nodes(3)

        '''
        self._raise_error_if_on_premise_ps()
        self._add_nodes(num_nodes)

    def replace_nodes(self, instance_ids):
        '''
        Replace each of the nodes specified in `instance_ids` with a new node.

        For each node terminated, add a new one in its place. After launching
        new nodes, terminate those specified by the `instance_ids` parameter.

        Parameters
        ----------
        instance_ids : list[str] | str
            The instance IDs for the nodes to terminate.

        See Also
        --------
        add_nodes, remove_nodes

        Notes
        -----
        Calling this method will clear out any existing data in the distributed
        cache.

        Examples
        --------
        To replace a node in the current Predictive Service with a new node:

          >>> print ps.get_status('node')
          +---------+-------------------------------+-------------+-----------+
          |  cache  |            dns_name           | instance_id |   state   |
          +---------+-------------------------------+-------------+-----------+
          | Healthy | ec2-12-11-12-121.us-west-2... |  i-4399eb8a | InService |
          | Healthy | ec2-21-21-24-221.us-west-2... |  i-ccc4663a | InService |
          +---------+-------------------------------+-------------+-----------+
          [2 rows x 4 columns]
          >>> ps.replace_nodes(['i-ccc4663a'])

        '''
        self._raise_error_if_on_premise_ps()

        if not isinstance(instance_ids, list):
            instance_ids = [instance_ids]

        if not all([type(i) in [str, unicode] for i in instance_ids]):
            raise TypeError('The "instance_ids" parameter must be either a ' \
                            'string or a list of strings')

        num_nodes_deployed = len(self._environment._get_all_hosts())
        num_replacements = len(instance_ids)

        if num_replacements == num_nodes_deployed:
            _logger.warn("Replacing all %d existing nodes in cluster" \
                         % num_replacements)

        # bring up replacement nodes
        _logger.info("Adding %d nodes to cluster", num_replacements)

        num_replaced = 0
        while num_replaced < num_replacements:
            n = min(num_replacements - num_replaced, _NODE_LAUNCH_LIMIT)
            self._add_nodes(n)
            num_replaced += n

        # terminate specified nodes
        self._remove_nodes(instance_ids)

        # try re-enabling cache
        self._manager_service.restart_cache()

        # check that cache is healthy, and warn if not
        expected_cache_status = "healthy" if self._global_cache_state else "disabled"
        if self._environment._is_cache_ok(expected_cache_status):
            _logger.info("Predictive service cluster is healthy")

    def terminate_service(self, remove_logs=False, remove_state=True):
        '''
        Terminates the Predictive Service deployment.

        This will terminate all EC2 hosts and delete the load balancer. The
        Predictive Service object is not usable after terminate_service is
        called.

        **This operation can not be undone.**

        Parameters
        ----------
        remove_logs : bool
            Delete all logs associated with this Predictive Service Deployment remotely.

        remove_state : bool
            Delete all state data associated with this Predictive Service Deployment remotely.
        '''
        # Terminate hosts and delete load balancer.
        self._raise_error_if_on_premise_ps()

        self._environment.terminate(remove_logs)

        # Remove state file, stored endpoints and dependencies
        if remove_state:
            self._environment.remove_state(self._state_path,
                                            PredictiveService._DEPENDENCIES_DIR,
                                            PredictiveService._PREDICTIVE_OBJECT_DIR)

    def _raise_error_if_on_premise_ps(self):
        '''
        Raise an exception if the environment is docker.
        '''
        import inspect
        if isinstance(self._environment, _DockerPredictiveServiceEnvironment) or \
               isinstance(self._environment, _LocalPredictiveServiceEnvironment):
            raise NotImplementedError("The function " + inspect.stack()[1][3] + " is disabled on the" +
                                     " client side for on-premises Predictive Service. Please check with your" +
                                     " systems admin for further instructions.")



    #
    # Metrics
    #

    def get_metrics_url(self):
        return self._manager_service.get_metrics_url()
    get_metrics_url.__doc__ = \
        _ManagerServiceClient.get_metrics_url.__doc__


    #
    # Logs
    #

    def get_log_path(self):
        return self._manager_service.get_logs()['path']
    get_log_path.__doc__ = _ManagerServiceClient.get_logs.__doc__

    def get_log_filenames(self, type, start_time=None, end_time=None):
        return self._manager_service.get_log(type,
            start_time=start_time,
            end_time=end_time)
    get_log_filenames.__doc__ = _ManagerServiceClient.get_log.__doc__


    def flush_logs(self):
        """
        Forces the service to push its logs to the log storage area. You can
        retrieve the log storage area with get_log_path
        """
        self._manager_service.flush_logs()

    #
    # Configuration
    #

    _all_cache_params = (
        'global_cache_state',
        'cache_max_memory_mb',
        'cache_ttl_on_update_secs',
        'cache_query_ttl_secs',
        'cache_feedback_ttl_secs',
    )
    _all_adhoc_execute_params = (
        'adhoc_execute',
    )
    _all_cors_params = (
        'cors_origin',
    )
    _all_scale_factor_params = (
        'scale_factor',
    )
    _all_params = (
        _all_cache_params
        + _all_adhoc_execute_params
        + _all_cors_params
        + _all_scale_factor_params)

    def get_configuration(self, *args):
        """Returns the configuration parameters for the Predictive Service.

        Parameters
        ----------

        *args : list of str

            The configuration parameters you are interested in. If specified,
            then only those parameters will be returned. If unspecified, all
            parameters are returned.

        Returns
        -------

        dict

            The keys are the configuration parameters and the values are the
            current configuration values.

            The meaning of the parameters are explained in ```configure```.

        See Also
        --------

        configure
        """
        if args:
            params = set(args)
        else:
            params = set(self._all_params)

        if not params.issubset(self._all_params):
            raise ValueError("Unexpected params: {}".format(
                ", ".join([repr(p) for p in params.difference(self._all_params)])
            ))

        result = {}

        cache_params = params.intersection(self._all_cache_params)
        if cache_params:
            system_cache_configuration = \
                self._manager_service.get_system_cache_configuration()

            for p in cache_params:
                result[p] = getattr(system_cache_configuration, p)


        adhoc_execute_params = params.intersection(self._all_adhoc_execute_params)
        if adhoc_execute_params:
            adhoc_execute_configuration = \
                self._manager_service.get_adhoc_execute_configuration()

            for p in adhoc_execute_params:
                result[p] = getattr(adhoc_execute_configuration, p)


        cors_params = params.intersection(self._all_cors_params)
        if cors_params:
            cors_configuration = \
                self._manager_service.get_cors_configuration()

            for p in cors_params:
                result[p] = getattr(cors_configuration, p)


        scale_factor_params = params.intersection(self._all_scale_factor_params)
        if scale_factor_params:
            scale_factor_configuration = \
                self._manager_service.get_scale_factor_configuration()

            for p in scale_factor_params:
                result[p] = getattr(scale_factor_configuration, p)


        return result


    def configure(self,
            global_cache_state=None,
            cache_max_memory_mb=None,
            cache_ttl_on_update_secs=None,
            cache_query_ttl_secs=None,
            cache_feedback_ttl_secs=None,
            adhoc_execute=None,
            cors_origin=None,
            scale_factor=None,
    ):
        """
        Updates specified system configuration parameters for the Predictive
        Service.

        Parameters
        ----------
        global_cache_state : str
            "enabled" or "disabled".

        cache_max_memory_mb : int
            The maximum memory the cache can use in MB.

        cache_ttl_on_update_secs : int
            How long the request to update the cache will wait before timing
            out in seconds.

        cache_query_ttl_secs : int
            How long the request to query the cache will wait before timing
            out in seconds.

        cache_feedback_ttl_secs : int
            How long the request to update the feedback cache will wait before
            timing out in seconds.

        adhoc_execute : str
            "enable" or "disable". Whether adhoc queries are allowed at all.
            Note that an admin key or an API key with adhoc_execute enabled is
            required in addition to this configuration otherwise adhoc
            executions cannot be performed.

        cors_origin : str
            The CORS origin for the server.

        scale_factor: int
            How many evaluators to run per node. Minimum of 1. Be careful not
            to set this too high or you will exhaust all of the memory on the
            system and it will become unstable.

        See Also
        --------

        get_configuration

        """
        params = {}

        if global_cache_state is not None:
            if global_cache_state not in ('enabled', 'disabled'):
                raise ValueError("global_cache_state must be 'enabled' or 'disabled'")
            params['global_cache_state'] = global_cache_state

        if cache_max_memory_mb is not None:
            cache_max_memory_mb = int(cache_max_memory_mb)
            if cache_max_memory_mb <= 0:
                raise ValueError("cache_max_memory_mb must be greater"
                    " than 0.")
            params['cache_max_memory_mb'] = cache_max_memory_mb

        if cache_ttl_on_update_secs is not None:
            cache_ttl_on_update_secs = int(cache_ttl_on_update_secs)
            if cache_ttl_on_update_secs < 0:
                raise ValueError("cache_ttl_on_update_secs must be greater"
                    " than 0.")
            params['cache_ttl_on_update_secs'] = cache_ttl_on_update_secs

        if cache_query_ttl_secs is not None:
            cache_query_ttl_secs = int(cache_query_ttl_secs)
            if cache_query_ttl_secs < 0:
                raise ValueError("cache_query_ttl_secs must be greater"
                    " than 0.")
            params['cache_query_ttl_secs'] = cache_query_ttl_secs

        if cache_feedback_ttl_secs is not None:
            cache_feedback_ttl_secs = int(cache_feedback_ttl_secs)
            if cache_feedback_ttl_secs < 0:
                raise ValueError("cache_feedback_ttl_secs must be greater"
                    " than 0.")
            params['cache_feedback_ttl_secs'] = cache_feedback_ttl_secs

        if adhoc_execute is not None:
            if adhoc_execute not in ('enable', 'disable'):
                raise ValueError("adhoc_execute must be 'enable' or 'disable'")
            params['adhoc_execute'] = adhoc_execute

        if cors_origin is not None:
            cors_origin = unicode(cors_origin)
            params['cors_origin'] = cors_origin

        if scale_factor is not None:
            scale_factor = int(scale_factor)

            if scale_factor < 1:
                raise ValueError("scale_factor cannot be less than 1.")

            params['scale_factor'] = scale_factor

        cache_params = set(self._all_cache_params).intersection(params)
        if cache_params:
            system_cache_configuration = \
                self._manager_service.get_system_cache_configuration()

            for p in cache_params:
                setattr(system_cache_configuration, p, params[p])

            self._manager_service.set_system_cache_configuration(system_cache_configuration)

        adhoc_execute_params = set(self._all_adhoc_execute_params).intersection(params)
        if adhoc_execute_params:
            adhoc_execute_configuration = \
                self._manager_service.get_adhoc_execute_configuration()

            for p in adhoc_execute_params:
                setattr(adhoc_execute_configuration, p, params[p])

            self._manager_service.set_adhoc_execute_configuration(adhoc_execute_configuration)

        cors_params = set(self._all_cors_params).intersection(params)
        if cors_params:
            cors_configuration = \
                self._manager_service.get_cors_configuration()

            for p in cors_params:
                setattr(cors_configuration, p, params[p])

            self._manager_service.set_cors_configuration(cors_configuration)

        scale_factor_params = set(self._all_scale_factor_params).intersection(params)
        if scale_factor_params:
            scale_factor_configuration = \
                self._manager_service.get_scale_factor_configuration()

            for p in scale_factor_params:
                setattr(scale_factor_configuration, p, params[p])

            self._manager_service.set_scale_factor_configuration(scale_factor_configuration)

    def repair(self):
        '''
        Repair management service in case the management server lost quorum.
        '''
        _logger.info("Trying to repair the consul cluster.")
        self._manager_service.consul_repair()
        _logger.info("Successfully repaired the consul cluster.")

    def _get_consul_info(self):
        '''
        Get information about the consul

        Returns
        -------
        A dictionary contains 'members', 'state' and 'info' regarding the consul
        '''
        return {
            'members' : self._manager_service.get_consul_members(),
            'info' : self._manager_service.get_consul_info(),
            'state' : self._manager_service.get_consul_state(),
        }
