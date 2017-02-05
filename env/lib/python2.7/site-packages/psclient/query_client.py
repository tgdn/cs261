"""
This module defines QueryClient that consumes service provided
by GraphLab Create Predictive Service.
"""

import os, os.path, sys
import json
import urllib
import requests
from ConfigParser import ConfigParser, NoOptionError
from requests.auth import HTTPBasicAuth

SERVICE_INFO_SECTION_NAME = "Service Info"
from .config import load_config


import logging as _logging
_logger = _logging.getLogger(__name__)

def connect(
        query_endpoint=None,
        api_key=None,
        admin_key=None,
        verify_certificate=None,
        config_file=None,
        name=None,
    ):
    """Loads a QueryClient object from config files or environment variables.

    The QueryClient is a thin client that does not do any of the cluster or
    management operations. It only knows about query() and feedback().

    The QueryClient needs query_endpoint and api_key to function. These can be
    specified directly, or they can be loaded from the environment variables
    or configuration files. verify_certificate is optional and defaults to
    False.

    The logic for determining the parameters is as follows. These steps are
    followed until all the parameters are found.

        * If the parameters are specified directly, use those.
        * If config_file or the environment variable $PS_CONFIG is specified:

            * If it's a file object, load directly from the object.
            * If it's a directory, look for <config_file>/<name>.conf.
            * If it's a file name, load from that.

        * Use the environment variables $PS_QUERY_ENDPOINT, $PS_API_KEY, or
        $PS_VERIFY_CERTIFICATE, if specified.
        * If config_file and $PS_CONFIG is not specified, then the following
        locations are searched for a configuration file:

            * <name>.conf in the current working directory.
            * <name>.conf in the directory of the script.
            * $XDG_CONFIG_HOME/<name>.conf, or ~/.local/share/<name>.conf if
            $XDG_CONFIG_HOME is not specified.
            * $XDG_CONFIG_DIRS/<name>.conf, or /etc/xdg/<name>.conf if
            $XDG_CONFIG_DIRS is not specified.
            * ~/.<name>.conf

    Config File Format
    ------------------

    The config file is standard windows or Python .ini format. It should look
    something like this:

        [Predictive Service]
        query_endpoint = https://.../
        api_key = 123...abc
        verify_certificate = True

    Configuration files can be generated with
    PredictiveService.generate_config(). One should also be generated when
    psclient.create() was successfully executed.


    Parameters
    ----------

    query_endpoint : str, optional

        The URL for the query endpoint.

    api_key : str, optional

        The API key. Note that API keys can be restricted only to work with
        certain endpoints via the PredictiveService object. If you use an
        admin key here, you should have access to all the endpoints.

    verify_certificate : bool, optional

        Whether to check the SSL certificate from the query client.

    name : str, optional

        The name of the service. This is used to find the configuration file.
        It will look for files named <name>.conf. By default, this is
        $PS_NAME, or if that is unspecified, 'psclient'.

    config_file : str or file, optional

        The configuration file to use. This can be a file object or it can be
        a path to a directory or a file. If it is a directory, then it looks
        for <name>.conf in that directory. If it is not specified, and the
        configuration parameters are not specified or found in the environment
        variables, then a series of locations will be searched for the
        appropriate configuration file.

    Returns
    -------

    QueryClient
    """

    params = load_config(
        name=name,
        config_file=config_file,
        params = {
            'query_endpoint':query_endpoint,
            'admin_key':admin_key,
            'api_key':api_key,
            'verify_certificate':verify_certificate,
        },
    )

    if params['query_endpoint'] is None:
        raise ValueError("query_endpoint was not specified")

    if params['api_key'] is None:
        if params['admin_key'] is None:
            raise ValueError("neither api_key nor admin_key were specified")
        params['api_key'] = params['admin_key']

    del params['admin_key']


    return QueryClient(**params)



class NonExistError(Exception):
    pass


class QueryClient(object):
    def __init__(self,
            query_endpoint=None,
            api_key=None,
            verify_certificate=None,
            query_timeout=None):

        if not query_endpoint or not api_key:
            raise ValueError("'query_endpoint' and 'api_key' need to be provided to"
                 " initialize QueryClient.")

        self.query_endpoint = query_endpoint
        self.api_key = api_key
        self.verify_certificate = verify_certificate or False

        if query_timeout is None:
            self.query_timeout = 10.0
        else:
            self.query_timeout = query_timeout
        self._ping()

    def __str__(self):
        '''String representation of the QueryClient'''
        s = ""
        s += 'Predictive Service Client:\n'
        s += "\tendpoint: %s\n" % self.query_endpoint
        return s

    def __repr__(self):
        '''String representation of the QueryClient'''
        return self.__str__()

    @property
    def query_timeout(self):
        """The timeout for queries in seconds."""
        return self._query_timeout

    @query_timeout.setter
    def query_timeout(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError('"query_timeout" value has to be a positive float in seconds.')

        self._query_timeout = value

    def query(self, uri, *args, **kwargs):
        '''Query a Predictive Service object

        Parameters
        ----------
        uri : str
            The model uri, must have been deployed in server side

        args : list | None
            The positional arguments passed into query method

        kwargs : dict | None
            The keyword arguments passed into query method

        Examples
        --------

            >>> client = QueryClient(config_file='some file')

            To predict a preference score for a product for a particular user:

            >>> data = {'dataset':{'user_id':175343, 'product_id':1011}}
            >>> client.query('recommender', method = 'predict', data = data)

            To predict preference scores for a list of user-product pairs:

            >>> data = {'dataset':[
                {'user_id':175343, 'product_id':1011},
                {'user_id':175344, 'product_id':1012}
                ]}
            >>> client.query('recommender', method='predict', data=data)
            >>> client.query('recommender', 'predict', data)

            To predictive preference scores:

            >>> client.query('recommender', method='predict', data=data)
            >>> client.query('recommender', 'predict', data)

        Returns
        -------
        out : dict
            Returns the query result.  If successful, the actual query result will
            be in result['response']

        '''
        if not isinstance(uri, basestring):
            raise TypeError("'uri' has to be a string or unicode")

        if args and kwargs:
            raise ValueError('Mixing of keyword arguments and positional arguments '
                                'when querying predictive service is not supported.')

        # convert to valid url
        uri = urllib.quote(uri)

        internal_data = {'data': args or kwargs}
        response = self._post(
            'query/%s' % uri,
            data=internal_data,
            timeout=self._query_timeout)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise NonExistError("Predictive Object '%s' cannot be found" % uri)
        else:
            raise RuntimeError('Query error status: %s, error: %s' % (response.status_code, response.text))

    def feedback(self, key, **kwargs):
        '''Provide feedback to the query result. This is a free format feedback.

        Parameters
        ----------
        key : str
            The UUID of the request.

        **kwargs : dict
            The data you would like to pass as part of the feedback.

        Returns
        -------
        out : dict
            Returns the server response.

        '''
        if not isinstance(key, basestring):
            raise RuntimeError("Expect key to be a string or unicode")

        data = {'data': kwargs, 'id': key}
        response = self._post('feedback', data=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError('Feedback error (%s): %s' % (
                response.status_code,
                response.text))

    def _ping(self):
        if not hasattr(self, 'session'):
            self.session = requests.session()

        _logger.info("Connecting to Predictive Service at %s" % self.query_endpoint)
        response = self.session.get(self.query_endpoint, verify=self.verify_certificate)
        if response.status_code == 200:
            _logger.info("Successfully connected to %s" % (self.query_endpoint))
            return response.text
        else:
            raise RuntimeError("Error responding from service: response: %s" % str(response.text))

    def _post(self, path, data, timeout = None):
        headers = {'content-type': 'application/json'}

        url = self.query_endpoint + '/' + path
        data = json.dumps(data)

        if not timeout or not isinstance(timeout, int):
            timeout = 10

        return self.session.post(url = url, data=data, headers=headers,\
               verify=self.verify_certificate, timeout=timeout,
               auth=HTTPBasicAuth('api_key', self.api_key))
