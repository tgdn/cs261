import random as _random
import copy as _copy
from datetime import datetime as _datetime
from time import time as _time
import sys as _sys
import collections as _collections
import graphlab as _graphlab
from graphlab import SFrame as _SFrame
from graphlab.toolkits.model_parameter_search._model_parameter_search_evaluators import default_evaluator as _default_evaluator
from graphlab.connect import _get_metric_tracker
from graphlab.toolkits.model_parameter_search._model_parameter_search import _check_if_sklearn_factory
from graphlab.toolkits.model_parameter_search._model_parameter_search import _combiner, _train_test_model
from graphlab.deps import numpy as _np
from graphlab.deploy import map_job as _map_job
from graphlab.toolkits.model_parameter_search._model_parameter_search import _add_docstring, _DATA_DOCSTRING, _MODEL_FACTORY_DOCSTRING, _MODEL_PARAMETERS_DOCSTRING, _EVALUATOR_DOCSTRING, _ENVIRONMENT_DOCSTRING, _RETURN_MODEL_DOCSTRING, _RETURNS_DOCSTRING


class KFold(object):
    """
    Create a K-Fold split of a data set as an iteratable/indexable
    object of K pairs, where each pair is a partition of the data set.

    This can be useful for cross validation, where each fold is used as a
    heldout dataset while training on the remaining data.

    Parameters
    ----------
    data : SFrame
      An SFrame containing the data.

    num_folds : int
      The number of folds to create.

    Notes
    -----
    This does not shuffle the data. Shuffling your data is a useful preprocessing
    step when doing cross validation for data assumed to be iid.

    Examples
    --------

    >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
    >>> data = gl.SFrame.read_csv(url)
    >>> data['label'] = (data['label'] == 'p')
    >>> folds = gl.cross_validation.KFold(data, 5)
    >>> for train, valid in folds:
            m = gl.boosted_trees_classifier.create(train, target='label')
            print m.evaluate(valid)
    # Get the i'th fold
    >>> (train, test) = folds[i]

    """

    def __init__(self, data, num_folds):

        _get_metric_tracker().track('cross_validation.KFold')

        if (data is not None):
            self.num_folds = num_folds
            self._split_sframe(data)
            self._cached_list = None

    @classmethod
    def from_list(cls, cached_fold_list):
        """
        Create a :class:`graphlab.toolkits.cross_validation.KFold` object
        directly from a list. Each element of this list must be a tuple
        where the first value is either a

            - SFrame
            - numpy array
            - str representing a path to a filename

        The second value of the tuple may either be the same type as the
        first or None.

        If strings are provided, the pathname can be one of the following:

            - **Locally**: Relative or absolute path.
            - **EC2**: An S3 path.
            - **Hadoop**: A full HDFS path including hostname.

        """
        _get_metric_tracker().track('cross_validation.KFold.from_list')

        ret = KFold(None, None)

        # Test type and materialize as list
        if not isinstance(cached_fold_list, _collections.Iterable):
            raise TypeError("Provided data folds must be list or iterable.")

        cached_fold_list = list(cached_fold_list)

        # If a list of single SFrames or arrays, convert to list of tuples
        # with None as second argument
        res = []
        for v in cached_fold_list:
            if not isinstance(v, tuple):
                if not (type(v) in [_SFrame, str] or isinstance(v, _np.ndarray)):
                    raise ValueError("If providing a list of singletons, each "
                    "value must be either a only SFrame, str, or numpy arrays")
                if isinstance(v, _np.ndarray):
                    res.append(((v, None), None))
                else:
                    res.append((v, None))
            else:
                if len(v) != 2:
                    raise ValueError("Only tuples of size 2 allowed as element "
                            "when constructing a KFold object from a list.")

                res.append(v)

        included_types = set()

        def _check_element_types(v):

            # Check the training element in the tuple
            if type(v[0]) in [_SFrame, str]:
                included_types.add("SFrame")

            elif isinstance(v[0], tuple):
                if (len(v[0]) != 2
                    or not isinstance(v[0][0], _np.ndarray)
                    or not (v[0][1] is None or isinstance(v[0][1], _np.ndarray))):

                    raise ValueError("Each training data set containing"
                        "numpy arrays must be a tuple of features and (optional) targets,"
                        " e.g. (X, y) or (X, None).")

                included_types.add("ndarray")

            else:
                raise ValueError("The first element of the tuple, e.g. the"
                                " training set, must be either a only SFrame, str,"
                                " or a tuple of numpy arrays")

            if v[1] is None:
                pass
            elif type(v[1]) in [_SFrame, str]:
                included_types.add("SFrame")
            elif isinstance(v[1], tuple):
                if (len(v[1]) != 2
                    or not isinstance(v[1][0], _np.ndarray)
                    or not (v[1][1] is None or isinstance(v[1][1], _np.ndarray))):

                    raise ValueError("Each validation data set containing"
                    "numpy arrays must be a tuple of features and (optional) targets,"
                    " e.g. (X, y), (X, None)")

                included_types.add("ndarray")
            else:
                raise ValueError("The second element of the tuple, e.g. the"
                            " training set, must be either a only SFrame, str,"
                            " a tuple of numpy arrays or must be None.")


        for v in res:
            _check_element_types(v)

        if included_types == {"ndarray", "SFrame"}:
            raise ValueError("Input data mixes Numpy arrays and SFrame types.")

        ret._cached_list = res
        ret.num_folds = len(cached_fold_list)
        return ret

    def __getitem__(self, fold_id):
        if self._cached_list is not None:
            return self._cached_list[fold_id]
        else:
            test = self.sframe_splits[fold_id]
            train = _SFrame()
            for i in range(self.num_folds):
                if i != fold_id:
                    train = train.append(self.sframe_splits[i])
            return train, test

    def __iter__(self):
        def generator():
            for j in range(self.num_folds):
                yield self.__getitem__(j)
        return generator()

    def _split_sframe(self, data):
        data.__materialize__()
        N = data.num_rows()
        fold_sizes = [N // self.num_folds] * self.num_folds
        for j in range(N % self.num_folds):
            fold_sizes[j] += 1

        start_location = 0
        self.sframe_splits = []
        for fold_size in fold_sizes:
            end_location = min(start_location + fold_size, N)
            self.sframe_splits.append(data[start_location:end_location])
            start_location = end_location


def shuffle(sf, random_seed=None):
    """
    Create a copy of the SFrame where the rows have been shuffled randomly.

    Parameters
    ----------
    sf : SFrame
      An SFrame of data.

    random_seed : int
      Random seed to use for the randomization. If provided, each call to this
      method will produce an identical result.

    Returns
    -------
    out : SFrame
      The same data, but the rows are in random order.

    Examples
    --------

    >>> sf = gl.SFrame({'feature': [0, 1, 2, 3, 4]})
    >>> x = gl.cross_validation.shuffle(sf)
    """
    _get_metric_tracker().track('cross_validation.shuffle')

    # Ensure the original data is not shuffled
    x = _copy.copy(sf)

    # Set random seed
    rng = _random.Random()
    if random_seed is not None:
        rng.seed(random_seed)

    # Create a new column name that doesn't exist
    cn = '_id'
    while cn in x.column_names():
        cn = '_' + cn

    # Sort by a column of random numbers
    if _sys.version_info.major == 3:
        _xrange = range
    else:
        _xrange = xrange
    x[cn] = [rng.random() for i in _xrange(x.num_rows())]
    x = x.sort(cn)
    del x[cn]

    return x

@_add_docstring(param_data=_DATA_DOCSTRING,
               param_model_factory=_MODEL_FACTORY_DOCSTRING,
               param_model_params=_MODEL_PARAMETERS_DOCSTRING,
               param_evaluator=_EVALUATOR_DOCSTRING,
               param_environment=_ENVIRONMENT_DOCSTRING,
               param_return_model=_RETURN_MODEL_DOCSTRING,
               param_returns=_RETURNS_DOCSTRING)
def cross_val_score(datasets,
                    model_factory,
                    model_parameters,
                    evaluator=_default_evaluator,
                    environment=None,
                    return_model=True):
    """
    Evaluate model performance via cross validation for a given set of
    parameters.

    Parameters
    ----------
    {param_data}
    {param_model_factory}

    model_parameters : dict
        The params argument takes a dictionary containing parameters that will
        be passed to the provided model factory.

    {param_evaluator}
    {param_environment}
    {param_return_model}
    {param_returns}

    See Also
    --------
    graphlab.toolkits.model_parameter_search.create

    Examples
    --------
    >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
    >>> data = gl.SFrame.read_csv(url)
    >>> data['label'] = (data['label'] == 'p')
    >>> folds = gl.cross_validation.KFold(data, 5)
    >>> params = dict([('target', 'label'), ('max_depth', 5)])
    >>> job = gl.cross_validation.cross_val_score(folds,
                                                  gl.boosted_trees_classifier.create,
                                                  params)
    >>> print job.get_results()
    """
    _get_metric_tracker().track('cross_validation.cross_val_score')

    if isinstance(datasets, _graphlab.SFrame):
        folds = [(datasets, None)]
    elif isinstance(datasets, tuple):
        if len(datasets) != 2:
            raise ValueError("Provided dataset tuple must be train/test pair.")
        folds = [datasets]
    else:
        folds = datasets

    if (not isinstance(folds, KFold)):
        folds = KFold.from_list(folds)

    num_folds = folds.num_folds
    include_fold_id = num_folds > 1


    params = []
    model_id = 0

    for fold_id in range(num_folds):

        metadata = {'model_id': model_id}
        if include_fold_id:
            metadata['fold_id'] = fold_id
        model_id += 1

        params.append({
            'model_factory': _check_if_sklearn_factory(model_factory, model_parameters),
            'model_parameters': model_parameters,
            'folds': folds,
            'evaluator': evaluator,
            'return_model': return_model,
            'metadata': metadata
        })
    now = _datetime.now().strftime('%b-%d-%Y-%H-%M-%S-%f')

    random_hash = str(hash( (id(folds), ("%.21f" % _time()) ) ) )[:8]

    job_name = "Cross-Validation-%s-%s" % (now, random_hash)

    return _map_job.create(_train_test_model,
                           parameter_set=params,
                           name=job_name,
                           environment=environment,
                           combiner_function=_combiner)
