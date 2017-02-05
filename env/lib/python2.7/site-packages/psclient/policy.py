import os as _os
import abc as _abc
from copy import copy as _copy
from random import random as _random, randrange as _randrange
from tempfile import mkdtemp as _mkdtemp

import pickle as _pickler
from psclient import file_util as _file_util
import shutil as _shutil
import logging as _logging
_logger = _logging.getLogger(__name__)

_SCHEMA_VERSION = 1

class EndpointPolicy(object):
    __metaclass__ = _abc.ABCMeta

    '''
    Base class for all model serving policy
    '''
    def __init__(self, models, values):
        if not isinstance(models, list):
            models = [models]

        if not hasattr(models, '__iter__') or \
            not all([(isinstance(m, basestring) and m != '') for m in models]):
            raise TypeError('"models" parameter has to be a list of non-empty string.')

        # All models cannot have the same name
        if len(set(models)) != len(models):
            raise ValueError('There are duplicate models in model list')

        if len(models) == 0:
            raise ValueError('The plicy has to have at least one model backing it.')

        if not values:
            values = [0 for i in range(len(models))]
        else:
            if not hasattr(values, '__iter__'):
                values = [values]

            if not all([(v <=1 and v >= 0) for v in values]):
                raise ValueError('"values" parameter has to be a list of floats between 0 and 1.')

        if len(models) != len(values):
            raise ValueError('"modes" and "values" do not have the same length')

        self._values = values
        self._models = models
        self.schema_version = _SCHEMA_VERSION

        # stats for model serving
        self._serve_counts = [0 for i in range(len(models))]
        self._feedback_counts = [0 for i in range(len(models))]

    def get_dependencies(self):
        return _copy(self._models)

    def get_state(self):
        '''
        Returns the state for the policy
        '''
        ret = {}
        for i in range(len(self._models)):
            ret[self._models[i]] = {
                'serve_count': self._serve_counts[i],
                'feedback_count': self._feedback_counts[i],
                'value': self._values[i],
            }
        return ret

    def get_models(self):
        '''
        Return a list of models that are served by this policy

        Returns
        -------
        models : list[str]
            A list of models that are served by this policy
        '''
        return _copy(self._models)

    def get_methods(self):
        raise NotImplementedError("In order to load the methods of the"
             " policy, you must first load the models and return one of their"
             " methods.")

    @_abc.abstractmethod
    def select_model(self, **kwargs):
        '''
        Select a model to serve

        Returns
        -------
        model_name : str
            The model identifier that is going to be served
        '''
        pass

    def feedback(self, model_name, **kwargs):
        '''
        Feedback to this policy so that the policy can potentially adjust over time

        Parameters
        ----------
        model_name : str
            The model to feedback

        kwargs : dict
            Feedback to the model

        Returns
        -------
        success : bool
            True if feedback has been processed successfully, False otherwise
        '''
        reward = kwargs.get('reward')
        try:
            if reward is not None and (type(reward) in [int, float]) \
                and reward <= 1 and reward >= 0:

                # This could raise ValueError exception if model_name is not in list
                model_index = self._models.index(model_name)

                self._feedback_counts[model_index] += 1
                n = self._feedback_counts[model_index]

                value = self._values[model_index]
                new_value = ((n - 1)/float(n)) * value + (1 / float(n)) * reward
                self._values[model_index] = new_value
                return True
            else:
                _logger.info('ignored feedback for model "%s" because invalid arguments: %s' % \
                    (model_name, kwargs))

        except Exception as e:
            _logger.warn('Ignore feedback because exception happend. %s' % e)

        return False

    @classmethod
    def load(cls, path, schema_version, aws_credentials={}):
        '''
        Load the policy from give path
        '''
        loaded_policy = None
        if (_file_util.is_local_path(path)):
            loaded_policy = cls._load_local(path)
        else:
            loaded_policy = cls._load_remote(path, schema_version, aws_credentials)

        return loaded_policy

    @classmethod
    def _load_local(cls, path):
        path = _file_util.expand_full_path(path)
        if not _os.path.exists(path):
            raise RuntimeError("Path %s does not exist." % path)

        try:
            with open(_os.path.join(path, 'policy'), 'r') as f:
                unpickler = _pickler.Unpickler(f)
                schema_version = unpickler.load()
                loaded_policy = unpickler.load()
                loaded_policy.schema_version = schema_version

        except Exception as e:
            raise RuntimeError('Unable to load policy. Error: %s' % e)

        return loaded_policy

    @classmethod
    def _load_remote(cls, path, schema_version, aws_credentials={}):
        temp_dir = _mkdtemp(prefix='predictive_policy_')

        if _file_util.is_s3_path(path):
            _file_util.download_from_s3(path, temp_dir, is_dir=True,
                              aws_credentials=aws_credentials, silent=True)
        elif _file_util.is_hdfs_path(path):
            _file_util.download_from_hdfs(path, temp_dir, is_dir=True)
        else:
            assert 'Only support S3 and HDFS path for Predictive Object saving location!'

        return cls._load_local(temp_dir)

    def save(self, path, aws_credentials = None):
        '''
        Persist the policy to a certain path
        '''
        if _file_util.is_local_path(path):
            self._save_local(path)
        else:
            self._save_remote(path, aws_credentials)

    def _save_remote(self, path, aws_credentials):
        tempdir = _mkdtemp(prefix='predictive_policy_')

        try:
            self._save_local(tempdir)
            if _file_util.is_s3_path(path):
                _file_util.upload_to_s3(tempdir, path, is_dir=True, \
                                        aws_credentials = aws_credentials)
            elif _file_util.is_hdfs_path(path):
                _file_util.hdfs_mkdir(path)
                _file_util.upload_to_hdfs(tempdir + '/*', path)
        finally:
            _shutil.rmtree(tempdir)

    def _save_local(self, path):
        '''Save current predictive object to local path
        '''
        if not _os.path.exists(path):
            _os.makedirs(path)

        with open(_os.path.join(path, 'policy'), 'wb') as f:
            pickler = _pickler.Pickler(f)
            schema_version = self.schema_version

            try:
                pickler.dump(schema_version)
                pickler.dump(self)
            except Exception as e:
                _logger.error('Unable to save object: %s' % (e.message))
                raise e

class EpsilonGreedyPolicy(EndpointPolicy):
    '''
    A model serving policy backed by epsilon-greedy algorithm that explores with
    a certain probability (defined by epsilon)

    Parameters
    ----------
    models : list[str] | str
        a list of models to be served by this policy

    epsilon : float between 0 and 1, optional
        The epsilon value for this policy. default value is 0.1

    values : list[float], optional
        Initial perforance values for the models, if not given, all 0s are used

    '''
    DEFAULT_EPSILON = 0.1

    def __init__(self, models, epsilon = DEFAULT_EPSILON, values = None):

        super(EpsilonGreedyPolicy, self).__init__(models, values)

        if epsilon > 1 or epsilon < 0:
            raise ValueError('"epsilon" has to be a value between 0 and 1.')
        self.epsilon = epsilon

    def select_model(self, **kwargs):
        '''
        Select a model to serve

        Returns
        -------
        model_name : str
            The model identifier that is going to be served
        '''
        r = _random()
        if r > self.epsilon:
            # exploitation
            max_value = max(self._values)

            # Randomly pick among those that have highest value
            indices = [i for i, x in enumerate(self._values) if x == max_value]
            r1 = _randrange(0, len(indices))
            return_index = indices[r1]
        else:
            # exploration
            return_index = _randrange(len(self._values))

        self._serve_counts[return_index] += 1
        return self._models[return_index]

    def get_doc_string(self):
        return 'Use Epsilon Greedy policy to choose model to evaluate.'

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "%s(%s, %s)" % (
                self.__class__.__name__,
                self._models, self.epsilon)


class ProbabilityPolicy(EndpointPolicy):
    '''
    A simple model selection policy that serves each model with some probability.

    Parameters
    ----------
    model_ratios : dict
        A dictionary specifies the serving ratio for each model for the A/B testng.
        The keys are the model uris, the values are the serving weight of
        each mode. The probability for each mode is the weight of the mode
        divided by the sum of all the weights.

    '''
    def __init__(self, model_ratios):
        if not isinstance(model_ratios, dict):
            raise TypeError("First parameter has to be a dictionary")

        models = model_ratios.keys()
        weights = model_ratios.values()

        super(ProbabilityPolicy, self).__init__(models, values = None)


        total_weight = sum(weights)
        self.ratios = [
            1.0 * weight / total_weight
            for weight in weights]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        ret_value = {}
        for (i,m) in enumerate(self._models):
            ret_value.update({
                m: self.ratios[i]
            })

        s = "%s: %s" % (self.__class__.__name__, str(ret_value))
        return s

    def get_doc_string(self):
        return 'Use simple probability policy to choose model to evaluate.'

    def select_model(self, **kwargs):
        '''
        Select a model to serve

        Returns
        -------
        model_name : str
            The model identifier that is going to be served
        '''
        r = _random()
        ratio_sum = 0

        # select last one by default
        return_index = len(self._models) - 1

        for (index, ratio) in enumerate(self.ratios):
            ratio_sum += ratio
            if r <= ratio_sum:
                return_index = index
                break

        self._serve_counts[return_index] += 1
        return self._models[return_index]
