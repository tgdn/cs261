"""
This module defines ModelPredictiveObject class that wraps a GraphLab Create model
into a Predictive Object that may be deployed into a Predictive Service
"""
from .predictive_object import PredictiveObject
from .file_util import is_path

import logging as _logging
_logger = _logging.getLogger(__name__)

import graphlab

class ModelPredictiveObject(PredictiveObject):
    '''Predictive Object definition for GraphLab Create models

    Each ModelPredictiveObject wraps one model and provides 'query' interface for
    the model
    '''
    def __init__(self, model, description = ''):
        super(ModelPredictiveObject, self).__init__(description)

        if not (isinstance(model, graphlab.Model) or \
                        isinstance(model, graphlab.CustomModel)) and not is_path(model):
                raise TypeError('Model must be a GraphLab Create model or a path to a model.')

        self._set_model(model)

    def query(self, method, data):
        '''Query the model according to input

        Query the model according to the method and query data specified in the
        input.

        Parameters
        ----------
        input : dict
            a dictionary that needs to have the following two keys:
                 'method': The method that is supported by given model. Refer to individual
                        model for list of supported methods.
                 'data' : the actual data that is used to query the model. In the form
                        of dictionary, which matches the actual method signature

        Returns:
        --------
        out: depends on model query output but must be JSON serializable.
            If model method returns SFrame, out will be converted to a list of dictionaries,
            if model method returns SArray, out will be converted to a list of values
        '''
        if method not in self._model_methods:
            raise ValueError("Method '%s' is not supported for current model" % method)

        if type(data) != dict:
            raise TypeError('"data" value has to be a dictionary')

        # do appropriate construction on SFrame or SArray depend on the method
        # definition. See _convert_to_SFrame and _convert_to_SArray to details.
        method_description = self._model_methods[method]
        for (param_name, param_types) in method_description.iteritems():
            if param_name not in data:
                continue

            value = data.get(param_name)
            converted = value  # no conversion by default

            for param_type in param_types:
                if param_type == 'sframe':
                    (success, converted) = self._convert_to_SFrame(value)
                elif param_type == 'sarray':
                    (success, converted) = self._convert_to_SArray(value)
                else:
                    raise RuntimeError("Unexpected type '%s', expected ('sframe', 'sarray')" % param_type)

                if success:
                    break

            if not success:
                raise TypeError("Cannot convert input '%s' from '%s' to '%s'. Only value of "\
                    " dictionary type or list of dictionary may be converted to 'sframe'." \
                    % (param_name, value, param_types))

            data.update({param_name: converted})

        # call actual method
        func = getattr(self.model.__class__, method)
        result = func(self.model, **data)

        # convert GraphLab Create object to python data for ease of serialization
        return self._make_serializable(result)

    @classmethod
    def _convert_to_SArray(cls, value):
        ''' Convert an input value to SArray, the logic is:

                list => an SArray of len(list) rows
                other => an SArray of one row

            Parameters
            ----------
            value : any type
                The value to be converted

            Returns
            -------
            (success, converted) : pair(bool, SArray | value)
                 'success' indicates if the conversion is successful,
                 if successful, 'converted' contains the converted value
                 otherwise, 'converted' is original value
        '''
        converted = value
        if not isinstance(converted, list):
            converted = [converted]

        # create an SArray now
        try:
            return (True, graphlab.SArray(converted))
        except Exception as e:
            logging.info("Hit exception trying to convert input %s to SArray. Error: %s" % (value, e.message))
            return (False, value)

    @classmethod
    def _convert_to_SFrame(cls, value):
        ''' Convert an input value to SFrame, the logic is:

                dict => an SFrame with one row, each key becomes a column
                list(dict) => an SFrame, each key becomes a column
                list(other) => no conversion
                other => no conversion

            Parameters
            ----------
            value : any type
                The value to be converted

            Returns
            -------
            (success, converted) : pair(bool, SFrame | value)
                 'success' indicates if the conversion is successful,
                 if successful, 'converted' contains the converted value,
                 otherwise, 'converted' is original value

        '''
        converted = value
        if isinstance(converted, dict):
            converted = [converted]

        if not hasattr(converted, '__iter__'):
            return (False, value)

        if not all([isinstance(v, dict) for v in converted]):
            return (False, value)

        # create an SFrame now
        try:
            pivoted_values = dict((x, []) for x in converted[0].keys())
            for row in converted:
                    for (k,v) in row.iteritems():
                            pivoted_values[k].append(v)
            return (True, graphlab.SFrame(pivoted_values))
        except Exception as e:
            logging.info("Hit exception trying to convert input %s to SFrame. Error: %s" % (converted, e.message))
            return (False, value)

    def _set_model(self, model):
        '''Extract supported methods from the model. Each model needs to implement
        a class method called

            _get_queryable_methods()

        that tells this Predictive Object whether or not it expects a SFrame, SArray
        or other type as input, the 'query' method of this class will automatically
        convert to appropriate SFrame or SArray that is needed. The model method can
        also expect either an SArray or an SFrame, for example, recommender.recommend()
        method could expect the first parameter 'user' to be either a list of users
        or an SFrame with more information regarding the users.

        For example, recommender model would return the following information:

                             {'predict': {
                                        'dataset': 'sframe',
                                        'new_observation_data': 'sframe',
                                        'new_user_data': 'sframe',
                                        'new_item_data': 'sframe'
                                },
                                'recommend': {
                                        'users': ['sframe', 'sarray'],
                                        'items': ['sframe', 'sarray'],
                                        'new_observation_data': 'sframe',
                                        'new_user_data': 'sframe',
                                        'new_item_data': 'sframe',
                                        'exclude': 'sframe'}
                                }
        '''
        if is_path(model):
            # This is a path, download the file and load it
            model = graphlab.load_model(model)

        self.model = model

        self._model_methods = model._get_queryable_methods()
        if hasattr(model, '_get_ml_metric_config'):
            self._ml_metric_config = model._get_ml_metric_config()
        else:
            self._ml_metric_config = {}

        if type(self._model_methods) != dict:
            raise RuntimeError("_get_queryable_methods for model %s should return a"
                "dictionary" % model.__class__)

        for (method, description) in self._model_methods.iteritems():
            if type(description) != dict:
                raise RuntimeError("model %s _get_queryable_methods should use dict as method"
                    "description." % model.__class__)

            for (param_name, param_types) in description.iteritems():
                # support either "sframe", "sarray" or ["sframe", "sarray"]
                if not isinstance(param_types, list):
                    param_types = [param_types]

                for param_type in param_types:
                    if (param_type not in ['sframe', 'sarray']):
                        raise RuntimeError("model %s _get_queryable_methods should only use"
                            "'sframe' or 'sarray' type. %s is not supported" % (model.__class__, param_type))

                description.update({param_name: param_types})

            self._model_methods.update({method: description})

    def get_ml_metric_config(self):
        '''Get the configuration that determines how feedback gets evaluated.
        '''
        if hasattr(self, '_ml_metric_config'):
            return self._ml_metric_config
        else:
            return {}

    def get_doc_string(self):
        '''Returns documentation for the predictive object query'''
        docstring_prefix = 'Note:\n'
        docstring_prefix += '    For input that expects "SFrame" type, you need to pass in a list of dictionaries,\n'
        docstring_prefix += '    for input that expects "SArray" type, you need to pass in a list of values.\n'
        docstring_prefix += '    Similarly, output of type SFrame will be converted to a list of dictionaries,\n'
        docstring_prefix += '    output of type SArray will be converted to a list of values.\n'
        docstring_prefix += '\n'
        docstring_prefix += 'The following methods are available for query for this predictive object:\n'
        docstring_prefix += '    %s' % (';'.join(self._model_methods))
        docstring_prefix += '\n'

        ret = docstring_prefix

        for method in self._model_methods:
            ret += '\n' + method + '\n'
            ret += getattr(self.model, method).__doc__

        return ret

    def get_methods(self):
        model_methods = self.model._get_queryable_methods()
        return [{'schema': {'data': None, 'sample': None, 'output': None}, 'method': method} for method in model_methods]
