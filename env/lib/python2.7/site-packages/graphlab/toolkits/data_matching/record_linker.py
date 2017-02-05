"""
Class definition and utilities for the nearest neighbors flavor of the record
linker.
"""

import copy as _copy
import time as _time
import array as _array

import graphlab as _gl
import graphlab.connect as _mt
from graphlab.toolkits._model import CustomModel as _CustomModel
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy

import graphlab.toolkits._internal_utils as _tkutl
from graphlab.toolkits._private_utils import _summarize_accessible_fields

from . import _util as _dmutl

from graphlab.toolkits.data_matching.nearest_neighbor_deduplication \
    import _construct_auto_distance

from graphlab.toolkits.data_matching.nearest_neighbor_deduplication \
    import _engineer_distance_features


def get_default_options():
    """
    Return information about options for the nearest neighbor version of
    deduplication.

    Returns
    -------
    out : SFrame
        Each row in the output SFrames correspond to a parameter, and includes
        columns for default values, lower and upper bounds, description ,and
        type.
    """
    out = _gl.SFrame({'name': ['distance', 'verbose'],
                      'default_value': [None, 'True'],
                      'parameter_type': ['', 'boolean'],
                      'lower_bound': [None, 0],
                      'upper_bound': [None, 1],
                      'description': ['Name of a distance function or a composite distance function.',
                                      'Progress printing flag.']})
    return out


def create(dataset, features=None, distance=None, method='auto', verbose=True,
           **kwargs):
    """
    Create a RecordLinker model to match query records to a reference dataset of
    records, assuming both sets have the same general form.

    Parameters
    ----------
    dataset : SFrame
        Reference data, against which to link new queries with the 'link'
        method. The 'dataset' SFrame must include at least the features
        specified in the 'features' or 'distance' parameter.

    features : list[string], optional
        Name of the columns with features to use in comparing records. 'None'
        (the default) indicates that all columns should be used. Each column can
        be one of the following types:

        - *Numeric*: values of numeric type integer or float.

        - *Array*: array of numeric (integer or float) values. Each array
          element is treated as a separate variable in the model.

        - *Dictionary*: key-value pairs with numeric (integer or float) values.
          Each key indicates a separate variable in the model.

        - *String*: string values.

        Please note: if 'distance' is specified as a composite distance, then
        that parameter controls which features are used in the model.

    distance : string or list[list], optional
        Function to measure the distance between any two input data rows. This
        may be one of two types:

        - *String*: the name of a standard distance function. One of
          'euclidean', 'squared_euclidean', 'manhattan', 'levenshtein',
          'jaccard', 'weighted_jaccard', 'cosine', 'dot_product' (deprecated),
          or 'transformed_dot_product'.

        - *Composite distance*: the weighted sum of several standard distance
          functions applied to various features. This is specified as a list of
          distance components, each of which is itself a list containing three
          items:

          1. list or tuple of feature names (strings)

          2. standard distance name (string)

          3. scaling factor (int or float)

        For more information about GraphLab Create distance functions, please
        see the :py:mod:`~graphlab.toolkits.distances` module.

        For sparse vectors, missing keys are assumed to have value 0.0.

        If 'distance' is left unspecified or set to 'auto', a composite distance
        is constructed automatically based on feature types.

    method : {'auto', brute_force', 'lsh', 'ball_tree'}, optional
        Strategy for the nearest neighbors search. If not specified or 'auto',
        the search strategy is chosen automatically based on the data type and
        dimension.

    verbose : bool, optional
        If True, print progress updates and model details.

    **kwargs : optional
        Options passed through to the nearest_neighbors toolkit for particular
        nearest neighbors search strategies:

        - *leaf_size*: for the ball tree method, the number of points in each
          leaf of the tree. The default is to use the max of 1,000 and n/(2^11),
          which ensures a maximum tree depth of 12.

        - *num_tables*: For the LSH method, the number of hash tables
          constructed.

        - *num_projections_per_table*: For the LSH method, the number of
          projections for each hash table.

    Returns
    -------
    out : RecordLinker model.

    See Also
    --------
    RecordLinker, graphlab.toolkits.nearest_neighbors

    Notes
    -----
    - Standardizing features is often a good idea with distance-based methods,
      but this model does *not* standardize features.

    - For features that all have the same type, the distance parameter may be a
      single standard distance function name (e.g. "euclidean"). In the model,
      however, all distances are first converted to composite distance
      functions; as a result, the 'distance' field in the model is always a
      composite distance.

    References
    ----------
    - Christen, Peter. "Data matching: concepts and techniques for record
      linkage, entity resolution, and duplicate detection." Springer Science &
      Business Media, 2012.

    Examples
    --------
    >>> homes = graphlab.SFrame({'sqft': [1230, 875, 1745],
    ...                          'street': ['phinney', 'fairview', 'cottage'],
    ...                          'city': ['seattle', 'olympia', 'boston'],
    ...                          'state': ['WA', 'WA', 'MA']})
    ...
    >>> model = graphlab.record_linker.create(homes, features=['city'],
    ...                                       distance='levenshtein')
    """

    _mt._get_metric_tracker().track('{}.create'.format(__name__))
    start_time = _time.time()


    ## Validate the 'dataset' input.
    _tkutl._raise_error_if_not_sframe(dataset, "dataset")
    _tkutl._raise_error_if_sframe_empty(dataset, "dataset")


    ## Clean the method options and create the options dictionary
    allowed_kwargs = ['leaf_size', 'num_tables', 'num_projections_per_table']
    _method_options = {}

    for k, v in kwargs.items():
        if k in allowed_kwargs:
            _method_options[k] = v
        else:
            raise _ToolkitError("'{}' is not a valid keyword argument".format(k) +
                                " for the nearest neighbors model. Please " +
                                "check for capitalization and other typos.")


    ## Validate the features input.
    if features is not None:
        if not hasattr(features, '__iter__'):
            raise TypeError("Input 'features' must be a list.")

        if not all([isinstance(x, str) for x in features]):
            raise TypeError("Input 'features' must contain only strings.")

    else:
        features = dataset.column_names()


    ## Validate and preprocess the distance input.
    col_types = {k: v for k, v in zip(dataset.column_names(),
                                      dataset.column_types())}

    if isinstance(distance, list):
        distance = _copy.deepcopy(distance)

    elif isinstance(distance, str):
        # this will likely produce errors downstream if 'features' was not
        # specified by the user.
        distance = [[features, distance, 1]]

    elif distance == None:
        distance = _construct_auto_distance(features, col_types)

    else:
        raise TypeError("Input 'distance' not understood. For the " +
                         "data matching toolkit, 'distance' must be a string or " +
                         "a composite distance list."   )


    ## Validate the composite distance and set it in the model.
    allowed_dists = {
        'euclidean': [int, float, _array.array],
        'squared_euclidean': [int, float, _array.array],
        'manhattan': [int, float, _array.array],
        'levenshtein': [str],
        'jaccard': [str, dict],
        'weighted_jaccard': [str, dict],
        'cosine': [int, float, str, dict, _array.array],
        'dot_product': [int, float, str, dict, _array.array],
        'transformed_dot_product': [int, float, str, dict, _array.array]}

    distance = _dmutl.validate_composite_distance(distance, row_label=None,
                                             allowed_dists=list(allowed_dists.keys()),
                                             verbose=verbose)


    ## Validate feauture types against distance functions.
    _dmutl.validate_distance_feature_types(dataset, distance, allowed_dists)


    ## Clean and impute string data.

    #  *** NOTE: after this, the composite distance and feature set will be
    #      modified and useless to the user, so set the state here. ***
    state = {'distance': distance,
             'num_distance_components': len(distance)}

    union_features = _dmutl.extract_composite_features(distance)

    _dataset = _copy.copy(dataset)
    _distance = _copy.deepcopy(distance)

    for ftr in union_features:
        if col_types[ftr] == str:
            new_ftr = '__clean.' + ftr
            _dataset[new_ftr] = _dataset[ftr].fillna("")
            _dataset[new_ftr] = _dataset[new_ftr].apply(
                lambda x: _dmutl.cleanse_string(x), dtype=str)

            for dist_comp in _distance:
                dist_comp[0] = [new_ftr if x == ftr else x for x in dist_comp[0]]


    ## Convert strings to dicts if the distance isn't levenshtein, and
    #  concatenate string columns within a distance component into a single
    #  feature.
    _dataset, _distance = _engineer_distance_features(_dataset, _distance)


    ## Create the nearest neighbors model and set in the model
    nn_model = _gl.nearest_neighbors.create(_dataset, distance=_distance,
                                             method=method, verbose=verbose,
                                             **kwargs)


    ## Postprocessing and formatting
    state.update({'verbose': verbose,
                  'num_examples': dataset.num_rows(),
                  'features': union_features,
                  'nearest_neighbors_model': nn_model,
                  'num_features': len(union_features),
                  'method': nn_model['method'],
                  'training_time': _time.time() - start_time})

    model = RecordLinker(state)
    return model


class RecordLinker(_CustomModel, _ProxyBasedModel):
    """
    The RecordLinker model uses a nearest neighbors search to
    find records in the reference dataset (set when the model is created) that
    are very similar to records passed to the 'link_records' method.

    This model should not be constructed directly. Instead, use
    :func:`graphlab.data_matching.record_linker.create` to create an instance of
    this model.
    """

    _PYTHON_NN_LINKER_MODEL_VERSION = 1

    def __init__(self, state={}):

        if 'nearest_neighbors_model' not in state:
            state['nearest_neighbors_model'] = None

        assert(isinstance(state['nearest_neighbors_model'],
                          _gl.nearest_neighbors.NearestNeighborsModel))

        self.__proxy__ = _PythonProxy(state)

    def _get_version(self):
        return self._PYTHON_NN_LINKER_MODEL_VERSION

    def _save_impl(self, pickler):
        """
        Save the model.

        The model is saved as a directory which can then be loaded using the
        :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        pickler : GLPickler
            An opened GLPickle archive (Do not close the archive.)

        See Also
        ----------
        graphlab.load_model

        Examples
        ----------
        >>> model.save('my_model_file')
        >>> loaded_model = graphlab.load_model('my_model_file')
        """
        _mt._get_metric_tracker().track(self.__module__ + '.save_impl')

        state = self.__proxy__
        pickler.dump(state)

    @classmethod
    def _load_version(self, unpickler, version):
        """
        Load a previously saved RecordLinker model.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handler.

        version : int
            Version number maintained by the class writer.
        """
        if version < 1:
            nn_model = unpickler.load()
            state = unpickler.load()
            state['nearest_neighbors_model'] = nn_model
            return RecordLinker(state)

        state = unpickler.load()
        return RecordLinker(state)

    def _get_summary_struct(self):
        """
        Return a structured description of the model, including (where relevant)
        the schema of the training data, description of the training data,
        training statistics, and model hyperparameters.

        Returns
        -------
        sections : list (of list of tuples)
            A list of summary sections. Each section is a list. Each item in a
            section list is a tuple of the form ('<label>','<field>').

        section_titles: list
            A list of section names. The order matches that of the 'sections'
            object.
        """
        model_fields = [
            ("Number of examples", 'num_examples'),
            ("Number of feature columns", 'num_features'),
            ("Number of distance components", 'num_distance_components'),
            ("Method", 'method')]

        training_fields = [('Total training time (seconds)', 'training_time')]

        return ([model_fields, training_fields], ["Schema", "Training"])

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """
        width = 36
        key_str = "{:<{}}: {}"

        (sections, section_titles) = self._get_summary_struct()
        accessible_fields = {
            "nearest_neighbors_model": "Model used internally to compute nearest neighbors."}

        out = _tkutl._toolkit_repr_print(self, sections, section_titles,
                                         width=width)
        out2 = _summarize_accessible_fields(accessible_fields, width=width)
        return out + "\n" + out2

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the RecordLinker model.
        """
        return self.__repr__()

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current model instance.
        """
        return {v: self.__proxy__[v] for v in get_default_options()['name']}

    @classmethod
    def _get_queryable_methods(cls):
        """
        Return a list of method names that are queryable from Predictive
        Services.

        Returns
        -------
        out : dict
            Dictionary of option and values used to train the current instance
            of the model.

        See Also
        --------
        get_default_options, list_fields, get
        """
        return {'link_records':{'dataset':'sframe'}}

    def link(self, dataset, k=5, radius=None, verbose=True):
        """
        Find matching records from the reference dataset (entered when the model
        was created) for each record in the 'dataset' passed to this function.
        The query dataset must include columns with the same names as the label
        and feature columns used to create the RecordLinker
        model.

        Parameters
        ----------
        dataset : SFrame
            Query data. Must contain columns with the same names and types as
            the features used to train the model. Additional columns are
            allowed, but ignored. Please see the nearest neighbors
            :func:`~graphlab.nearest_neighbors.create` documentation for more
            detail on allowable data types.

        k : int, optional
            Maximum number of nearest neighbors to return from the reference set
            for each query observation. The default is 5, but setting it to
            ``None`` will return all neighbors within ``radius`` of the query
            point.

        radius : float, optional
            Only neighbors whose distance to a query point is smaller than this
            value are returned. The default is ``None``, in which case the ``k``
            nearest neighbors are returned for each query point, regardless of
            distance.

        verbose : bool, optional
            If True, print progress updates and model details.

        Returns
        -------
        out : SFrame
            An SFrame with the k-nearest neighbors of each query observation.
            The result contains four columns: the first is the row label of the
            query observation, the second is the row label of the nearby
            reference observation, the third is the distance between the query
            and reference observations, and the fourth is the rank of the
            reference observation among the query's k-nearest neighbors.

        Notes
        -----
        - If both ``k`` and ``radius`` are set to ``None``, each query point
          returns all of the reference set. If the reference dataset has
          :math:`n` rows and the query dataset has :math:`m` rows, the output is
          an SFrame with :math:`nm` rows.

        Examples
        --------
        Assume we've created the model from the example in the RecordLinker
        'create' function.

        >>> queries = graphlab.SFrame({'sqft': [986, 1320],
        ...                            'street': ['fremont', 'phiney'],
        ...                            'city': ['sea', 'seattle'],
        ...                            'state': ['WA', 'WA']})
        ...
        >>> model.link(queries, k=2, radius=5.)
        +-------------+-----------------+----------+------+
        | query_label | reference_label | distance | rank |
        +-------------+-----------------+----------+------+
        |      0      |        0        |   4.0    |  1   |
        |      0      |        2        |   5.0    |  2   |
        |      1      |        0        |   0.0    |  1   |
        +-------------+-----------------+----------+------+
        """
        _mt._get_metric_tracker().track(self.__module__ + '.link_records')

        ## Validate the 'dataset' input.
        _tkutl._raise_error_if_not_sframe(dataset, "dataset")
        _tkutl._raise_error_if_sframe_empty(dataset, "dataset")

        ## Make sure all of the necessary features are present at 'link' time.
        sf_features = _tkutl._toolkits_select_columns(dataset,
                                                      self.get('features'))

        ## Clean and impute string data. *** Think about consolidating this and
        #  the next step into a feature transformer.***
        col_types = {k: v for k, v in zip(dataset.column_names(),
                                          dataset.column_types())}
        _dataset = _copy.copy(dataset)
        _distance = _copy.deepcopy(self.__proxy__['distance'])

        for ftr in self.get('features'):
            if col_types[ftr] == str:
                new_ftr = '__clean.' + ftr
                _dataset[new_ftr] = _dataset[ftr].fillna("")
                _dataset[new_ftr] = _dataset[new_ftr].apply(
                    lambda x: _dmutl.cleanse_string(x), dtype=str)

                for dist_comp in _distance:
                    dist_comp[0] = [new_ftr if x == ftr else x for x in dist_comp[0]]


        ## Convert strings to dicts and concatenate string features.
        _dataset, _ = _engineer_distance_features(_dataset, _distance)


        ## Query the nearest neighbor model
        result = self.__proxy__['nearest_neighbors_model'].query(_dataset, k=k, radius=radius,
                                      verbose=verbose)
        return result
