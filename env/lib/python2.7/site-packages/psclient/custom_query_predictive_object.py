"""
This module defines the CustomQueryPredictiveObject class that allows wrapping of
one or many GraphLab Create data structures and additional functionality within a
predictive object, which can be deployed into a Predictive Service.
"""

import types as _types
from .predictive_object import PredictiveObject as _PredictiveObject

import logging as _logging
_logger = _logging.getLogger(__name__)

class CustomQueryPredictiveObject(_PredictiveObject):
    def __init__(self, query, description = '', schema = None):
        '''Create a new CustomQueryPredictiveObject.

        Parameters
        -----------

        query : function
            Function that defines a custom query method. The query can have any
            signature, but input and output of the query needs to be JSON serializable.

        description : str
            The description of the custom predictive object

        '''
        super(CustomQueryPredictiveObject, self).__init__(description)

        self.custom_query = query
        self.required_packages = \
            query.required_packages if hasattr(query, 'required_packages') else []
        self.required_files = \
            query.required_files if hasattr(query, 'required_files') else {}

        self.schema = schema
        self._validate_parameters(query, self.required_packages, description,
                                  self.schema)
        if hasattr(query, '_get_ml_metric_config'):
            self._ml_metrics_config = query._get_ml_metric_config()
        else:
            self._ml_metrics_config = {}

    def get_ml_metric_config(self):
        '''Get the configuration that determines how feedback gets evaluated.
        '''
        return self._ml_metrics_config

    def query(self, *args, **kwargs):
        '''Query the custom defined query method using the given input.

        Parameters
        ----------
        args : list
            positional arguments to the query

        kwargs : dict
            keyword arguments to the query

        Returns
        -------
        out: object.
            The results depends on the implementation of the query method.
            Typically the return value will be whatever that function returns.
            However if it returns an SFrame, the SFrame will be automatically
            converted to a list of dicts. If it returns an SArray, the SArray
            will be converted to a list.

        See Also
        --------
        PredictiveObject, ModelPredictiveObject
        '''
        # include the dependent files in sys path so that the query can run correctly

        try:
            ret = self.custom_query(*args, **kwargs)
        except Exception as e:
            _logger.info('Exception hit when running custom query, error: %s' % e.message)
            raise e

        try:
            return self._make_serializable(ret)
        except Exception as e:
            _logger.info('Cannot properly serialize custom query result, error: %s' % e.message)
            raise e

    def get_doc_string(self):
        '''Get doc string from customized query'''
        if self.custom_query.__doc__ is not None:
            return self.custom_query.__doc__
        else:
            return "-- no docstring found in query function --"

    @classmethod
    def _validate_parameters(cls, query, required_packages, description,
                             schema):
        if not isinstance(query, _types.FunctionType):
            raise TypeError('Query parameter has to be a function')

        # validate python required packages, should be a list of strings
        assert (isinstance(required_packages, list))
        assert (all([isinstance(dependency, str) for dependency in required_packages]))

        if not isinstance(description, str):
            raise TypeError("'description' has to be a string")

        assert (isinstance(schema, dict) or schema is None)
        if isinstance(schema, dict):
            assert (isinstance(schema['input'], dict) or schema['input'] is None)
            assert (isinstance(schema['sample'], dict) or schema['sample']is None)
            assert (isinstance(schema['output'], dict) or schema['output'] is None)

    def get_methods(self):
        return [self.get_query_method()]

    def get_query_method(self):
        return {'method': 'query',
                'schema' : self.schema}
