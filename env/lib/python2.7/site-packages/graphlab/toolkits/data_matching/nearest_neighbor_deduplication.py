"""
Class definition and utilities for the nearest neighbor flavor of deduplication.
"""

import graphlab as _gl
import graphlab.connect as _mt
import copy as _copy
import time as _time
import array as _array
import logging as _logging
from graphlab.toolkits.data_matching.deduplication import _Deduplication
from . import _util as _dmutl
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _raise_error_if_sframe_empty
from graphlab.toolkits._internal_utils import _raise_error_if_not_sframe
from graphlab.toolkits._private_utils import _summarize_accessible_fields
from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy


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
    out = _gl.SFrame({'name': ['k', 'radius', 'verbose']})
    out['default_value'] = ['3', None, 'True']
    out['description'] = [
        'Number of neighbors to consider as matches for each point',
        'Maximum allowed distance between a point and a match',
        'Verbose printing']
    out['lower_bound'] = [1, 0, 0]
    out['upper_bound'] = [None, None, 1]
    out['parameter_type'] = ['integer', 'float', 'boolean']

    return out


def _construct_auto_distance(features, column_types):
    """
    Construct a composite distance function for a set of features, based on the
    types of those features.

    NOTE: This function is very similar to
    `:func:_nearest_neighbors._construct_auto_distance`. The function is
    separate in GLC v1.3 because it works for string distance names, and because
    the auto-distance logic for deduplication might be different than for
    general nearest neighbors.

    Parameters
    ----------
    features : list[str]
        Names of for which to construct a distance function.

    column_types : dict(string, type)
        Names and types of all columns.

    Returns
    -------
    dist : list[list]
        A composite distance function. Each element of the inner list has three
        elements: a list of feature names (strings), a distance function name
        (string), and a weight (float).

    See Also
    --------
    graphlab.toolkits.nearest_neighbors._construct_auto_distance
    """

    ## Put input features into buckets based on type.
    numeric_ftrs = []
    dict_ftrs = []

    for ftr in features:
        try:
            ftr_type = column_types[ftr]
        except:
            raise ValueError("The specified feature does not exist in the " +
                             "input data.")

        if ftr_type == str or ftr_type == dict:
            dict_ftrs.append(ftr)

        elif ftr_type in [int, float]:
            numeric_ftrs.append(ftr)

        else:
            raise TypeError("Unable to automatically construct a distance " +
                            "function for feature '{}'.".format(ftr))

    ## Construct the distance function
    dist = []

    if len(numeric_ftrs) > 0:
        dist.append([numeric_ftrs, 'euclidean', len(numeric_ftrs)])

    if len(dict_ftrs) > 0:
        dist.append([dict_ftrs, 'weighted_jaccard', len(dict_ftrs)])

    return dist


def _engineer_distance_features(dataset, distance):
    """
    Transform columns of the input dataset according to the distance/feature
    combinations specified in a composite distance list. For 'levenshtein'
    distance, multiple string columns are concatenated; for 'jaccard',
    'weighted_jaccard', 'cosine', 'dot_product' (deprecated), and
    'transformed_dot_product' distances, string features are transformed into
    counts of 3-character shingles.

    Parameters
    ----------
    dataset : SFrame
        Input dataset. Must contain columns with the feature names specified in
        ``distance``.

    distance : composite_distance
        Specification of a linear combination of features and distance functions
        for comparing records on those features.

    Returns
    -------
    dataset : SFrame
        The input ``dataset``, with additional columns for transformed features.

    clean_dist : composite_distance
        The input ``distance`` with ``label`` removed from feature lists and
        transformed feature names.
    """

    clean_dist = []

    for i, (comp_ftrs, comp_dist, weight) in enumerate(distance):

        if comp_dist == 'levenshtein' and len(comp_ftrs) > 1:

            new_ftr, dataset = _dmutl.concat_string_features(dataset=dataset,
                                                              features=comp_ftrs,
                                                              prefix='__concat.')
            clean_dist.append([[new_ftr], 'levenshtein', distance[i][2]])

        elif comp_dist in ['jaccard', 'weighted_jaccard', 'cosine',
                           'dot_product', 'transformed_dot_product']:
            new_ftrs, dataset = _dmutl.string_features_to_dict(dataset=dataset,
                                                               features=comp_ftrs,
                                                               prefix='__dict.')
            clean_dist.append([new_ftrs, distance[i][1], distance[i][2]])

        # just replaces the feature list with a clean list
        else:
            clean_dist.append([comp_ftrs, distance[i][1], distance[i][2]])

    return dataset, clean_dist


def create(datasets, row_label=None, features=None, grouping_features=None,
           distance=None, k=2, radius=None, verbose=True):
    """
    Create a deduplication model based on nearest neighbors and SGraph connected
    components.

    This method creates a :class:`NearestNeighborDeduplication` model by
    constructing a nearest neighbors similarity graph on all of the rows in the
    input 'datasets', then using the connected components tool in the
    :mod:`~graphlab.toolkits.graph_analytics` module to assign an entity label
    to each record. Records which share the same label are considered to be
    duplicates.

    .. warning::

        The 'dot_product' distance is deprecated and will be removed in future
        versions of GraphLab Create. Please use 'transformed_dot_product'
        distance instead, although note that this is more than a name change; it
        is a *different* transformation of the dot product of two vectors.
        Please see the distances module documentation for more details.

    Parameters
    ----------
    datasets : SFrame or list[SFrame] or dict(string: SFrame)
        Input datasets. Each SFrame in the list must include all of the features
        specified in the `features` or 'distance' parameters, but may
        have additional columns as well. SFrames can be input as values in a
        dictionary, where the keys are strings used in the output to identify
        the SFrame from which each record originated.

    row_label : string, optional
        Name of the SFrame column with row labels. If not specified, row numbers
        are used to identify rows in the output.

    features : list[string], optional
        Name of the columns with features to use in comparing records. 'None'
        (the default) indicates the intersection of columns over all SFrames in
        `datasets` should be used (except the label column, if specified). Each
        column can be one of the following types:

        - *Numeric*: values of numeric type integer or float.

        - *Array*: array of numeric (integer or float) values. Each array
          element is treated as a separate variable in the model.

        - *Dictionary*: key-value pairs with numeric (integer or float) values.
          Each key indicates a separate variable in the model.

        - *String*: string values.

        Please note: if `distance` is specified as a composite distance, then
        that parameter controls which features are used in the model. Any
        additional columns named in 'features' will be included in the model
        output but not used for distance computations.

    grouping_features : list[string], optional
        Names of features to use in grouping records before finding approximate
        matches. These columns must have string or integer type data. See the
        Notes section for more details on grouping.

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

    k : int, optional
        Number of neighbors to consider for each point.

    radius : float, optional
        Maximum distance from each point to a potential duplicate.

    verbose : bool, optional
        If True, print progress updates and model details.

    Returns
    -------
    out : NearestNeighborDeduplication model
        The NearestNeighborDeduplication object contains a field 'entities'
        which shows the entity label for each input record. It also shows the
        features for each record that are used to construct the model, as well
        as the original SFrame and row label for each record. If the original
        `datasets` are passed in a list, the SFrame identifier is the index of
        the SFrame in that list.

    See Also
    --------
    NearestNeighborDeduplication, graphlab.toolkits.nearest_neighbors,
    graphlab.SFrame.groupby

    Notes
    -----
    - Standardizing features is often a good idea with distance-based methods,
      but this model does *not* standardize features.

    - For datasets with more than about 10,000 records, *grouping* (also known
      as *blocking*) is a critical step to avoid computing distances between all
      pairs of records. The grouping step simply assigns each record to a group
      that has identical values for all `grouping_features`, and only looks for
      duplicates within each group.

    - Records with missing data in the `grouping_features` are removed from
      consideration as duplicates. These records are given the entity label
      "None".

    - For tasks that require *only* exact matches on certain features, it is
      generally more natural to use the SFrame `groupby` function.

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
    >>> sf1 = graphlab.SFrame({'id': [0, 1, 2],
    ...                        'x0': [0.5, 0.5, 0.3],
    ...                        'x1': [1., 0.8, 0.6],
    ...                        'city': ['seattle', 'olympia', 'boston'],
    ...                        'state': ['WA', 'WA', 'MA']})
    ...
    ... # note: misspellings in the following dataset do not prevent correct
    ... # matches.
    >>> sf2 = graphlab.SFrame({'id': [9, 10],
    ...                        'x0': [0.35, 0.4],
    ...                        'x1': [0.65, 0.8],
    ...                        'city': ['bostan', 'seatle'],
    ...                        'state': ['MA', 'WA']})
    ...
    >>> dist = [[('city',), 'levenshtein', 2],
    ...         [('x0', 'x1'), 'euclidean', 1.5]]
    ...
    >>> m = graphlab.nearest_neighbor_deduplication.create({'a': sf1, 'b': sf2},
    ...                                                    row_label='id',
    ...                                                    grouping_features=['state'],
    ...                                                    distance=dist, k=None,
    ...                                                    radius=3)
    ...
    >>> print m['entities']
    +----------+----+----------+-------+------+---------+------+
    | __sframe | id | __entity | state |  x0  |   city  |  x1  |
    +----------+----+----------+-------+------+---------+------+
    |    a     | 1  |    0     |   WA  | 0.5  | olympia | 0.8  |
    |    a     | 0  |    1     |   WA  | 0.5  | seattle | 1.0  |
    |    b     | 10 |    1     |   WA  | 0.4  |  seatle | 0.8  |
    |    a     | 2  |    2     |   MA  | 0.3  |  boston | 0.6  |
    |    b     | 9  |    2     |   MA  | 0.35 |  bostan | 0.65 |
    +----------+----+----------+-------+------+---------+------+
    [5 rows x 7 columns]
    """

    ## Set up
    _mt._get_metric_tracker().track('{}.create'.format(__name__))
    start_time = _time.time()

    model = NearestNeighborDeduplication()
    model.__proxy__['verbose'] = verbose
    model.__proxy__['k'] = k
    model.__proxy__['radius'] = radius


    ### ----------------------------- ###
    ### Validation and preprocessing ###
    ### ----------------------------- ###

    ### Validate input datasets
    ### -----------------------

    ## If datasets is already a dict, check the keys are all strings
    if isinstance(datasets, dict):
        if not(all([isinstance(x, str) for x in datasets.keys()])):
            raise ValueError("Keys in the 'datasets' dict must be strings.")

    ## Convert singleton SFrame dataset into a list of datasets
    if isinstance(datasets, _gl.SFrame):
        _raise_error_if_sframe_empty(datasets, "dataset")
        datasets = {0: datasets}

    ## Convert a list of SFrames into a dict
    if isinstance(datasets, list):
        datasets = {k: sf for k, sf in enumerate(datasets)}

    ## At this point, 'datasets' must be dict. If it's not, something is wrong.
    if not isinstance(datasets, dict):
        raise TypeError("Input 'datasets' must be an SFrame, a list of SFrames, " +
                        "or a dictionary of (string, SFrame) pairs.")

    model.__proxy__['num_datasets'] = len(datasets)

    ## Ensure that all datasets are SFrames
    for d in datasets.values():
        _raise_error_if_not_sframe(d, "dataset")


    ### Validate row label
    ### ------------------

    ## Validate the label column
    if row_label:
        if not isinstance(row_label, str):
            raise TypeError("The 'row_label' parameter must be the name (string " +
                            "type) of a column in each of the input datasets.")

        for d in datasets.values():
            if row_label not in d.column_names():
                raise _ToolkitError("The specified row_label column does not " +
                                    " exist in all input datasets.")
    else:
        row_label = 'row_number'

        for d in datasets.values():
            if row_label in d.column_names():
                raise _ToolkitError("Input 'row_label' defaulted to " +
                                    "'row_number', which is already a column" +
                                    " in at least one input dataset. Please " +
                                    "specify a row label column manually.")

    model.__proxy__['row_label'] = row_label


    ### Validate 'features' and 'grouping_features' parameters
    ### ------------------------------------------------------
    if features is not None:
        if not hasattr(features, '__iter__'):
            raise TypeError("Input 'features' must be a list.")

        if not all([isinstance(x, str) for x in features]):
            raise TypeError("Input 'features' must contain only strings.")

    if grouping_features is not None:
        if not hasattr(grouping_features, '__iter__'):
            raise TypeError("Input 'grouping_features' must be a list.")

        if not all([isinstance(x, str) for x in grouping_features]):
            raise TypeError("Input 'grouping_features' must contain only strings.")


    ### Validate and preprocess the distance function
    ### ---------------------------------------------
    # - The form of the 'distance' controls how we interact with the 'features'
    #   parameter as well.

    ## Find the intersection of all feature sets and feature types
    col_types = {k: v for k, v in zip(list(datasets.values())[0].column_names(),
                                      list(datasets.values())[0].column_types())}

    all_features = [sf.column_names() for sf in datasets.values()]
    ftr_intersection = list(set(all_features[0]).intersection(*all_features))
    ftr_intersection = [x for x in ftr_intersection if x != row_label]


    ## Convert features and distance arguments into a composite distance.
    if isinstance(distance, list):
        distance = _copy.deepcopy(distance)

    elif isinstance(distance, str):
        if features is not None:
            distance = [[features, distance, 1]]
        else:
            distance = [[ftr_intersection, distance, 1]]

    elif distance == None:
        if features is not None:
            distance = _construct_auto_distance(features, col_types)
        else:
            distance = _construct_auto_distance(ftr_intersection, col_types)

    else:
        raise TypeError("Input 'distance' not understood. Note that for the " +
                         "data matching toolkit, 'distance' must be a string or " +
                         "a composite distance list."   )


    ## Validate the form of the composite distance and add to the model
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

    distance = _dmutl.validate_composite_distance(distance, row_label,
                                                  list(allowed_dists.keys()),
                                                  verbose)
    model.__proxy__['distance'] = _copy.deepcopy(distance)


    ## Figure out which features are 'fuzzy', i.e. used for approximate
    #  matching, and set in the model state.
    fuzzy_features = _dmutl.extract_composite_features(distance)  # already has row_label removed

    model.__proxy__['features'] = fuzzy_features
    model.__proxy__['num_features'] = len(fuzzy_features)


    ## Compile a master list of all features. This includes grouping features,
    #  fuzzy features (the ones used for approximate matching), and "ancillary"
    #  features, which are specified in the 'features' parameter but not in the
    #  composite distance function for whatever reason. by the user in the
    #  'features' parameter, but not included in the 'distance' specification
    #  for some reason.
    if features is None:
        features = []
    else:
        features = [x for x in features if x != row_label]

    if grouping_features is None:
        grouping_features = []
    else:
        grouping_features = [x for x in grouping_features if x != row_label]

    model.__proxy__['grouping_features'] = grouping_features
    model.__proxy__['num_grouping_features'] = len(grouping_features)

    master_features = list(set(features + grouping_features + fuzzy_features))


    ### Consolidate data and engineer features
    ### --------------------------------------

    ## Consolidate multiple input datasets into a single SFrame, with a useful
    #  row label.
    sf_union = _dmutl.concatenate_sframes(datasets, row_label=row_label,
                                   features=master_features,
                                   sf_index_name='__sframe')
    overall_label = '__sframe.' + row_label
    sf_union[overall_label] = (sf_union['__sframe'].astype(str) + "." +
                               sf_union[row_label].astype(str))


    ## Validate the feature types in the consolidated dataset against the
    #  specified distance functions.
    _dmutl.validate_distance_feature_types(sf_union, distance, allowed_dists)


    ## Clean string-type features in the fuzzy feature set.
    for ftr in fuzzy_features:
        if col_types[ftr] == str:
            new_ftr = '__clean.' + ftr
            sf_union[new_ftr] = sf_union[ftr].fillna("")
            sf_union[new_ftr] = sf_union[new_ftr].apply(
                lambda x: _dmutl.cleanse_string(x), dtype=str)

            for dist_comp in distance:
                dist_comp[0] = [new_ftr if x == ftr else x for x in dist_comp[0]]


    ## Feature engineering, distance-component-wise. Also update list of
    #  features and a map to their types.
    sf_union, distance = _engineer_distance_features(sf_union, distance)
    transformed_features = _dmutl.extract_composite_features(distance)

    ### -------------------------------------------- ###
    ### Main loop over blocks of neighbor candidates ###
    ### -------------------------------------------- ###

    ## Construct blocks on features that must match exactly
    if verbose:
        _logging.info("Constructing groups of records that match exactly on " +
                      "the 'grouping_features'.")

    sf_union, block_errors, blocks = \
        _dmutl.construct_exact_blocks(sf_union, grouping_features)

    if verbose and len(distance) > 0 and blocks['Count'].max() > 10000:
        _logging.warning("There are more than 10,000 records in the largest match " +
            "group. For many uses, approximate matches within each match group are " +
            "computed with brute force nearest neighbors, which may be slow. " +
            "Consider using smaller groups by requiring different features to " +
            "match exactly.")

    max_entity_number = 0
    sf_entity = _gl.SFrame()
    output_features = (master_features + [row_label, '__sframe', '__entity'])

    ## Main loop over blocks
    for i, block in enumerate(blocks):

        if verbose:
            _logging.info("Processing {} records in match group: {}/{}".format(block['Count'],
                                                                         i+1,
                                                                         len(blocks)))

        ## Retrieve records in the block and impute the mean for missing numeric
        #  values.
        records = sf_union[block['min_idx']:(block['max_idx'] + 1)]
        complete_records = _dmutl.impute_numeric_means(records, transformed_features)

        if len(distance) > 0:
            ## Run all-point nearest neighbors
            if verbose:
                _logging.info("Building the similarity graph....")

            m = _gl.nearest_neighbors.create(complete_records, label=overall_label,
                                             distance=distance, verbose=False)
            knn = m.query(complete_records, label=overall_label, k=k, radius=radius,
                          verbose=verbose)


            ## Construct similarity graph to resolve transitive closure
            sg = _gl.SGraph()
            sg = sg.add_vertices(records[[overall_label]], vid_field=overall_label)
            sg = sg.add_edges(knn, src_field='query_label',
                              dst_field='reference_label')


            ## Cut the similarity graph to establish an entity for each vertex
            if verbose:
                _logging.info("Finding duplicate records in the similarity graph....")

            cc = _gl.connected_components.create(sg, verbose=verbose)

            ## Relabel the component IDs to be consecutive integers starting with
            #  the max index of the previous block's entity labels.
            block_labels = cc['component_size'].add_row_number('__entity')
            block_labels['__entity'] += max_entity_number
            max_entity_number += block_labels.num_rows()
            block_entity_labels = cc['component_id'].join(block_labels,
                                                          on='component_id',
                                                          how='left')

            ## Join the entity labels for the block back to the block's records,
            #  then append to the master output
            records = records.join(block_entity_labels[['__id', '__entity']],
                                   on={overall_label: '__id'}, how='left')
            records = records.sort('__entity')

        else:  # no fuzzy features, so no nearest neighbors, just block ID
            records['__entity'] = _gl.SArray.from_const(i, len(records))


        sf_entity = sf_entity.append(records[output_features])


    ### ------------------------------------- ###
    ### Postprocessing and results formatting ###
    ### ------------------------------------- ###

    ## Add rows missing from the blocking back to the master results
    if len(block_errors) > 0:
        block_errors['__entity'] = _gl.SArray.from_const(None, len(block_errors)).astype(int)
        sf_entity = sf_entity.append(block_errors[output_features])

    ## Rearrange columns
    sf_entity.swap_columns('__sframe', sf_entity.column_names()[0])
    sf_entity.swap_columns(row_label, sf_entity.column_names()[1])
    sf_entity.swap_columns('__entity', sf_entity.column_names()[2])


    ## Finalize the model state
    model.__proxy__['training_time'] = _time.time() - start_time
    model.__proxy__['entities'] = sf_entity
    model.__proxy__['num_entities'] = max_entity_number

    return model


class NearestNeighborDeduplication(_Deduplication, _ProxyBasedModel):
    """
    The NearestNeighborDeduplication model finds the nearest neighbors for each
    record in all datasets, and uses this information to build a similarity
    graph on the data. Entities are determined by finding the connected
    components in this graph.

    This model should not be constructed directly. Instead, use
    :func:`graphlab.data_matching.deduplication.create` or
    :func:`graphlab.data_matching.nearest_neighbor_deduplication.create` to
    create an instance of this model.
    """
    _PYTHON_NN_DEDUPE_MODEL_VERSION = 0

    def __init__(self, state=None):
        if state is None:
            state = {}

        self.__proxy__ = _PythonProxy(state)

    def _get_version(self):
        return self._PYTHON_NN_DEDUPE_MODEL_VERSION

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
        A function to load a previously saved NearestNeighborDeduplication
        model.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handler.

        version : int
            Version number maintained by the class writer.
        """
        _mt._get_metric_tracker().track(self.__module__ + '.load_version')

        state = unpickler.load()
        return NearestNeighborDeduplication(state)

    def _get_summary_struct(self):
        """
        Returns a structured description of the model, including (where relevant)
        the schema of the training data, description of the training data,
        training statistics, and model hyperparameters.

        Returns
        -------
        sections : list (of list of tuples)
            A list of summary sections.
              Each section is a list.
                Each item in a section list is a tuple of the form:
                  ('<label>','<field>')
        section_titles: list
            A list of section names.
              The order matches that of the 'sections' object.
        """

        model_fields = [
            ('Number of input datasets', 'num_datasets'),
            ('Number of feature columns', 'num_features'),
            ('Number of neighbors per point (k)', 'k'),
            ('Max distance to a neighbor (radius)', 'radius'),
            ('Number of entities', 'num_entities'),
            ('Total training time (seconds)', 'training_time')
        ]

        training = []
        training.append(('Total training time (seconds)', 'training_time'))

        return ([model_fields, training], ["Schema", "Training"])

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """
        width = 36
        key_str = "{:<{}}: {}"

        (sections, section_titles) = self._get_summary_struct()
        accessible_fields = {
            "entities": "Consolidated input records plus entity labels."}

        out = _toolkit_repr_print(self, sections, section_titles, \
                                                            width=width)
        out2 = _summarize_accessible_fields(accessible_fields, width=width)
        return out + "\n" + out2

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current NearestNeighborDeduplication instance.

        Returns
        -------
        out : dict
            Dictionary of option and values used to train the current instance
            of the NearestNeighborDeduplication.

        See Also
        --------
        get_default_options, list_fields, get
        """
        return {v: self.__proxy__[v] for v in get_default_options()['name']}
