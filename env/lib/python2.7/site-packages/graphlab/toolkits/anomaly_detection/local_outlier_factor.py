"""
Class definition and utilities for the Local Outlier Factor tool.
"""

import time as _time
import logging as _logging
from array import array as _array
import graphlab as _gl
import graphlab.connect as _mt
from graphlab.toolkits._model import CustomModel as _CustomModel
import graphlab.toolkits._internal_utils as _tkutl
from graphlab.toolkits._private_utils import _summarize_accessible_fields
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy

def get_default_options():
    """
    Information about local outlier factor parameters.

    Returns
    -------
    out : SFrame
        Each row in the output SFrames correspond to a parameter, and includes
        columns for default values, lower and upper bounds, description, and
        type.
    """
    out = _gl.SFrame({
        'name': ['distance', 'num_neighbors', 'threshold_distances',
                 'verbose'],
        'default_value': ['None', '5', 'True', 'True'],
        'parameter_type': ['String, function, or composite distance', 'int',
                           'bool', 'bool'],
        'lower_bound': ['None', '1', 'False', 'False'],
        'upper_bound': ['None', 'None', 'True', 'True'],
        'description': ['Name of a distance function or a composite distance function.',
                        'Number of neighbors to consider for each point.',
                        'Whether computed distances should be thresholded.',
                        'Progress printing flag.']})
    return out


def create(dataset, features=None, label=None, distance=None, num_neighbors=5,
           threshold_distances=True, verbose=True):
    """
    Create a :class:`LocalOutlierFactorModel`. This mode contains local outlier
    factor (LOF) scores for the training data passed to this model, and can
    predict the LOF score for new observations.

    The LOF method scores each data instance by computing the ratio of the
    average densities of the instance's neighbors to the density of the
    instance itself. The higher the score, the more likely the instance is to
    be an outlier *relative to its neighbors*. A score of 1 or less means that
    an instance has a density similar (or higher) to its neighbors and is
    unlikely to be an outlier.

    The model created by this function contains an SFrame called 'scores' that
    contains the computed local outlier factors. The `scores` SFrame has four
    columns:

        - *row_id*: the row index of the instance in the input dataset. If a
          label column is passed, the labels (and the label name) are passed
          through to this column in the output.

        - *density*: the density of instance as estimated by the LOF
          procedure.

        - *neighborhood_radius*: the distance from the instance to its
          furthest neighbor (defined by 'num_neighbors', and used for
          predicting the LOF for new points).

        - *anomaly_score*: the local outlier factor.

    For more information on the LOF method and the computation used for each of
    these columns, please see the Notes and References sections below.

    Parameters
    ----------
    dataset : SFrame
        Input dataset. The 'dataset' SFrame must include the features specified
        in the 'features' or 'distance' parameter (additional columns are
        ignored).

    features : list[string], optional
        Names of feature columns. 'None' (the default) indicates that all
        columns should be used. Each column can be one of the following types:

        - *Numeric*: values of numeric type integer or float.

        - *Array*: array of numeric (integer or float) values. Each array
          element is treated as a separate variable in the model.

        - *Dictionary*: key-value pairs with numeric (integer or float) values.
          Each key indicates a separate variable in the model.

        - *String*: string values.

        Please note: if 'distance' is specified as a composite distance, then
        that parameter controls which features are used in the model. Also note
        that the column of row labels is automatically removed from the
        features, if there is a conflict.

    label : str, optional
        Name of the input column containing row labels. The values in this
        column must be integers or strings. If not specified, row numbers are
        used by default.

    distance : string or list[list], optional
        Function to measure the distance between any two input data rows. If
        left unspecified, a distance function is automatically constructed
        based on the feature types. The distance may be specified by either a
        string or composite distance:

        - *String*: the name of a standard distance function. One of
          'euclidean', 'squared_euclidean', 'manhattan', 'levenshtein',
          'jaccard', 'weighted_jaccard', 'cosine', or 'dot_product'. Please see
          the :mod:`distances` module for more details.

        - *Composite distance*: the weighted sum of several standard distance
          functions applied to various features. This is specified as a list of
          distance components, each of which is itself a list containing three
          items:

          1. list or tuple of feature names (strings)

          2. standard distance name (string)

          3. scaling factor (int or float)

    num_neighbors : int, optional
        Number of neighbors to consider for each point.

    threshold_distances : bool, optional
        If True (the default), the distance between two points is thresholded.
        This reduces noise and can improve the quality of results, but at the
        cost of slower computation. See the notes below for more detail.

    verbose : bool, optional
        If True, print progress updates and model details.

    Returns
    -------
    model : LocalOutlierFactorModel
        A trained :class:`LocalOutlierFactorModel`, which contains an SFrame
        called 'scores' that includes the 'anomaly score' for each input
        instance.

    See Also
    --------
    LocalOutlierFactorModel, graphlab.toolkits.nearest_neighbors

    Notes
    -----
    - The LOF method scores each data instance by computing the ratio of the
      average densities of the instance's neighbors to the density of the
      instance itself. According to the LOF method, the estimated density of a
      point :math:`p` is the number of :math:`p`'s neighbors divided by the sum
      of distances to the instance's neighbors. In the following, suppose
      :math:`N(p)` is the set of neighbors of point
      :math:`p`, :math:`k` is the number of points in this set (i.e.
      the 'num_neighbors' parameter), and :math:`d(p, x)` is the distance
      between points :math:`p` and :math:`x` (also based on a user-specified
      distance function).

      .. math:: \hat{f}(p) = \\frac{k}{\sum_{x \in N(p)} d(p, x)}

    - The LOF score for point :math:`p` is then the ratio of :math:`p`'s
      density to the average densities of :math:`p`'s neighbors:

      .. math:: LOF(p) = \\frac{\\frac{1}{k} \sum_{x \in N(p)} \hat{f}(x)}{\hat{f}(p)}

    - If the 'threshold_distances' flag is set to True, exact distances are
      replaced by "thresholded" distances. Let  Suppose :math:`r_k(x)` is the
      distance from :math:`x` to its :math:`k`'th nearest neighbor. Then the
      thresholded distance from point :math:`p` to point :math:`x_i` is

      .. math:: d^*(p, x) = \max\{r_k(x), d(p, x)\}

      This adaptive thresholding is used in the original LOF paper (see the
      References section) to reduce noise in the computed distances and improve
      the quality of the final LOF scores.

    - For features that all have the same type, the distance parameter may be a
      single standard distance function name (e.g. "euclidean"). In the model,
      however, all distances are first converted to composite distance
      functions; as a result, the 'distance' field in the model is always a
      composite distance.

    - Standardizing features is often a good idea with distance-based methods,
      but this model does *not* standardize features.

    - If there are several observations located at an identical position, the
      LOF values can be undefined. An LOF score of "nan" means that a point is
      either in or near a set of co-located points.

    - This implementation of LOF forces the neighborhood of each data instance
      to contain exactly 'num_neighbors' points, breaking ties arbitrarily.
      This differs from the original LOF paper (see References below), which
      allows neighborhoods to expand if there are multiple neighbors at exactly
      the same distance from an instance.

    References
    ----------
    - Breunig, M. M., Kriegel, H., Ng, R. T., & Sander, J. (2000). `LOF:
      Identifying Density-Based Local Outliers
      <http://people.cs.vt.edu/badityap/classes/cs6604-Fall13/readings/breunig-2000.pdf>`_,
      pp 1-12.

    Examples
    --------
    >>> sf = graphlab.SFrame({'x0': [0., 1., 1., 0., 1., 0., 5.],
    ...                       'x1': [2., 1., 0., 1., 2., 1.5, 2.5]})
    >>> lof = graphlab.local_outlier_factor.create(sf, num_neighbors=3)
    >>> lof['scores']
    +--------+----------------+----------------+---------------------+
    | row_id |    density     | anomaly_score  | neighborhood_radius |
    +--------+----------------+----------------+---------------------+
    |   0    | 0.927050983125 | 1.03785526045  |         1.0         |
    |   3    | 0.962144739546 | 0.919592692017 |         1.0         |
    |   1    | 0.765148090776 | 1.14822979837  |         1.0         |
    |   6    | 0.230412599692 | 3.52802012342  |    4.71699056603    |
    |   2    | 0.71140803489  | 1.26014768739  |    1.80277563773    |
    |   5    | 0.962144739546 | 0.919592692017 |    1.11803398875    |
    |   4    | 0.962144739546 | 0.919592692017 |    1.11803398875    |
    +--------+----------------+----------------+---------------------+
    [7 rows x 4 columns]
    """

    ## Start the training time clock and instantiate an empty model
    _mt._get_metric_tracker().track(
        'toolkit.anomaly_detection.local_outlier_factor.create')

    logger = _logging.getLogger(__name__)
    start_time = _time.time()

    ## Validate the input dataset
    _tkutl._raise_error_if_not_sframe(dataset, "dataset")
    _tkutl._raise_error_if_sframe_empty(dataset, "dataset")


    ## Validate the number of neighbors, mostly to make the error message use
    #  the right parameter name.
    if not isinstance(num_neighbors, int):
        raise TypeError("Input 'num_neighbors' must be an integer.")

    if num_neighbors <= 0:
        raise ValueError("Input 'num_neighbors' must be larger than 0.")

    if num_neighbors > dataset.num_rows():
        num_neighbors = dataset.num_rows()

        if verbose:
            logger.info("Input 'num_neighbors' is larger than the number " +
                        "of rows in the input 'dataset'. Resetting " +
                        "'num_neighbors' to the dataset length.")

    ## Validate the row label against the features *using the nearest neighbors
    #  tool with only one row of data. This is a hack - we should encapsulate
    #  the validation steps in nearest neighbors and do them here first.
    validation_model = _gl.nearest_neighbors.create(dataset[:1], label=label,
                                                    features=features,
                                                    distance=distance,
                                                    method='brute_force',
                                                    verbose=False)

    ## Compute the similarity graph based on k and radius, without self-edges,
    #  but keep it in the form of an SFrame. Do this *without* the row label,
    #  because I need to sort on the row number, and row labels that aren't
    #  already in order will be screwed up.
    knn_model = _gl.nearest_neighbors.create(dataset,
                                             distance=validation_model.distance,
                                             method='brute_force',
                                             verbose=verbose)

    knn = knn_model.similarity_graph(k=num_neighbors, radius=None,
                                     include_self_edges=False,
                                     output_type='SFrame',
                                     verbose=verbose)

    ## Bias the distances by making them at least equal to the *reference*
    #  point's k'th neighbor radius. This is "reach-distance" in the original
    #  paper.
    if threshold_distances is True:
        radii = knn.groupby('query_label',
                        {'neighborhood_radius': _gl.aggregate.MAX('distance')})

        knn = knn.join(radii, on={'reference_label': 'query_label'},
                       how='left')

        knn['distance'] = knn.apply(
            lambda x: x['distance'] if x['distance'] > x['neighborhood_radius'] \
                 else x['neighborhood_radius'])


    ## Find the sum of distances from each point to its neighborhood, then
    #  compute the "local reachability density (LRD)". This is not remotely a
    #  valid density estimate, but it does have the form of mass / volume,
    #  where the mass is estimated by the number of neighbors in point x's
    #  neighborhood, and the volume is estimated by the sum of the distances
    #  between x and its neighbors.
    #
    ## NOTE: if a vertex is co-located with all of its neighbors, the sum of
    #  distances will be 0, in which case the inverse distance sum value is
    #  'inf'.
    scores = knn.groupby('query_label',
                         {'dist_sum': _gl.aggregate.SUM('distance')})

    scores['density'] = float(num_neighbors) / scores['dist_sum']


    ## Join the density of each point back to the nearest neighbors results,
    #  then get the average density of each point's neighbors' densities.
    knn = knn.join(scores, on={'reference_label': 'query_label'},
                   how='left')

    scores2 = knn.groupby('query_label',
                    {'average_neighbor_density': _gl.aggregate.AVG('density')})

    ## Combine each point's density and average neighbor density into one
    #  SFrame, then compute the local outlier factor (LOF).
    scores = scores.sort('query_label')
    scores2 = scores2.sort('query_label')
    scores['anomaly_score'] = scores2['average_neighbor_density'] / scores['density']


    ## Add each point's neighborhood radius to the output SFrame.
    if threshold_distances is True:
        radii = radii.sort('query_label')
        scores['neighborhood_radius'] = radii['neighborhood_radius']


    ## Remove the extraneous columns from the output SFrame and format.
    scores = scores.remove_column('dist_sum')


    ## Substitute in the row labels.
    if label is None:
        row_label_name = 'row_id'
        scores = scores.rename({'query_label': row_label_name})

    else:
        row_label_name = label
        scores = scores.remove_column('query_label')
        col_names = scores.column_names()
        scores[row_label_name] = dataset[label]
        scores = scores[[row_label_name] + col_names]


    ## Post-processing and formatting
    state = {
        'nearest_neighbors_model': knn_model,
        'verbose': verbose,
        'threshold_distances': threshold_distances,
        'num_neighbors': num_neighbors,
        'num_examples': dataset.num_rows(),
        'distance': knn_model['distance'],
        'num_distance_components': knn_model['num_distance_components'],
        'features': knn_model['features'],
        'row_label_name': row_label_name,
        'num_features': knn_model['num_features'],
        'unpacked_features': knn_model['unpacked_features'],
        'num_unpacked_features': knn_model['num_unpacked_features'],
        'scores': scores,
        'training_time': _time.time() - start_time}

    model = LocalOutlierFactorModel(state)
    return model


class LocalOutlierFactorModel(_CustomModel, _ProxyBasedModel):
    """
    Local outlier factor model. The LocalOutlierFactorModel contains the local
    outlier factor scores for training data passed to the 'create' function, as
    well as a 'predict' method for scoring new data. Outliers are determined by
    comparing the probability density estimate of each point to the density
    estimates of its neighbors.

    This model should not be constructed directly. Instead, use
    :func:`graphlab.anomaly_detection.create` or
    :func:`graphlab.anomaly_detectcion.local_outlier_factor.create` to create
    an instance of this model.

    Please see the API docs for the ``create`` method, as well as the
    references below or the `Anomaly Detection chapter of the User Guide <https://turi.com/learn/userguide/anomaly_detection/intro.html>`_ for more information on the Local Outlier Factor method.

    See Also
    --------
    create

    References
    ----------
    - Breunig, M. M., Kriegel, H., Ng, R. T., & Sander, J. (2000). `LOF:
      Identifying Density-Based Local Outliers
      <http://people.cs.vt.edu/badityap/classes/cs6604-Fall13/readings/breunig-2000.pdf>`_,
      pp 1-12.

    Examples
    --------
    >>> sf = graphlab.SFrame({'x0': [0., 1., 1., 0., 1., 0., 5.],
    ...                       'x1': [2., 1., 0., 1., 2., 1.5, 2.5]})
    >>> lof = graphlab.local_outlier_factor.create(sf, num_neighbors=3)
    >>> lof['scores']
    +--------+----------------+----------------+---------------------+
    | row_id |    density     | anomaly_score  | neighborhood_radius |
    +--------+----------------+----------------+---------------------+
    |   0    | 0.927050983125 | 1.03785526045  |         1.0         |
    |   3    | 0.962144739546 | 0.919592692017 |         1.0         |
    |   1    | 0.765148090776 | 1.14822979837  |         1.0         |
    |   6    | 0.230412599692 | 3.52802012342  |    4.71699056603    |
    |   2    | 0.71140803489  | 1.26014768739  |    1.80277563773    |
    |   5    | 0.962144739546 | 0.919592692017 |    1.11803398875    |
    |   4    | 0.962144739546 | 0.919592692017 |    1.11803398875    |
    +--------+----------------+----------------+---------------------+
    [7 rows x 4 columns]
    """

    _PYTHON_LOF_MODEL_VERSION = 2

    def __init__(self, state={}):

        if 'nearest_neighbors_model' not in state:
            state['nearest_neighbors_model'] = None
        if state['nearest_neighbors_model'] and not isinstance(state['nearest_neighbors_model'],
               _gl.nearest_neighbors._nearest_neighbors.NearestNeighborsModel):
            raise TypeError("The internal nearest neighbors model for LocalOutlierFactorModel is not correct.")

        self.__proxy__ = _PythonProxy(state)

    def _get_version(self):
        return self._PYTHON_LOF_MODEL_VERSION

    def _save_impl(self, pickler):
        """
        Save the model as a directory, which can be loaded with the
        :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        pickler : GLPickler
            An opened GLPickle archive (Do not close the archive).

        See Also
        --------
        graphlab.load_model

        Examples
        --------
        >>> model.save('my_model_file')
        >>> loaded_model = graphlab.load_model('my_model_file')
        """
        state = self.__proxy__
        pickler.dump(state)

    @classmethod
    def _load_version(self, unpickler, version):
        """
        Load a previously saved LocalOutlierFactorModel instance.

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
        else:
            state = unpickler.load()

        if version < 2:
            state['row_label_name'] = 'row_id'

        return LocalOutlierFactorModel(state)

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the LocalOutlierFactorModel.
        """
        return self.__repr__()

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """

        width = 40
        key_str = "{:<{}}: {}"

        sections, section_titles = self._get_summary_struct()
        accessible_fields = {
            "scores": "Local outlier factor for each row in the input dataset.",
            "nearest_neighbors_model": "Model used internally to compute nearest neighbors."}

        out = _tkutl._toolkit_repr_print(self, sections, section_titles,
                                         width=width)
        out2 = _summarize_accessible_fields(accessible_fields, width=width)
        return out + "\n" + out2

    def _get_summary_struct(self):
        """
        Returns a structured description of the model, including (where
        relevant) the schema of the training data, description of the training
        data, training statistics, and model hyperparameters.

        Returns
        -------
        sections : list (of list of tuples)
            A list of summary sections.
              Each section is a list.
                Each item in a section list is a tuple of the form:
                  ('<label>','<field>')
        section_titles: list
            A list of section titles.
              The order matches that of the 'sections' object.
        """
        model_fields = [
            ('Number of examples', 'num_examples'),
            ('Number of feature columns', 'num_features'),
            ('Number of neighbors', 'num_neighbors'),
            ('Use thresholded distances', 'threshold_distances'),
            ('Number of distance components', 'num_distance_components'),
            ('Row label name', 'row_label_name')]

        training_fields = [
            ('Total training time (seconds)', 'training_time')]

        section_titles = ['Schema', 'Training summary']
        return([model_fields, training_fields], section_titles)

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current LocalOutlierFactorModel instance.
        """
        return {k: self.__proxy__[k] for k in get_default_options()['name']}

    def predict(self, dataset, verbose=True):
        """
        Compute local outlier factors for new data. The LOF scores for new data
        instances are based on the neighborhood statistics for the data used
        when the model was created. Each new point is scored independently.

        Parameters
        ----------
        dataset : SFrame
            Dataset of new points to score with LOF against the training data
            already stored in the model.

        verbose : bool, optional
            If True, print progress updates and model details.

        Returns
        -------
        out : SArray
            LOF score for each new point. The output SArray is sorted to match
            the order of the 'dataset' input to this method.

        Examples
        --------
        >>> sf = graphlab.SFrame({'x0': [0., 1., 1., 0., 1., 0., 5.],
        ...                       'x1': [2., 1., 0., 1., 2., 1.5, 2.5]})
        >>> m = graphlab.local_outlier_factor.create(sf, num_neighbors=3)
        ...
        >>> sf_new = graphlab.SFrame({'x0': [0.5, 4.5],
        ...                           'x1': [1., 4.0]})
        >>> m.predict(sf_new)
        dtype: float
        Rows: 2
        [0.9317508614964032, 2.905646339288692]
        """
        _mt._get_metric_tracker().track(
            'toolkit.anomaly_detection.local_outlier_factor.predict')

        ## Validate the input dataset
        _tkutl._raise_error_if_not_sframe(dataset, "dataset")
        _tkutl._raise_error_if_sframe_empty(dataset, "dataset")

        num_neighbors = self.__proxy__['num_neighbors']

        ## Query the knn model with the new points.
        knn = self.__proxy__['nearest_neighbors_model'].query(dataset, k=num_neighbors, verbose=verbose)

        ## Join the reference data's neighborhood statistics to the nearest
        #  neighbors results.
        knn = knn.join(self.__proxy__['scores'], on={'reference_label': 'row_id'},
                       how='left')

        # Compute reachability distance for each new point and its
        # neighborhood.
        if self.__proxy__['threshold_distances'] is True:
            knn['distance'] = knn.apply(
                lambda x: x['distance'] \
                    if x['distance'] > x['neighborhood_radius'] \
                    else x['neighborhood_radius'])

        ## Find the sum of distances from each point to its neighborhood, then
        #  compute the "local reachability density" for each query point.
        scores = knn.groupby('query_label',
                             {'dist_sum': _gl.aggregate.SUM('distance')})

        scores['density'] = float(num_neighbors) / scores['dist_sum']


        ## Find the average density for each query point's neighbors.
        scores2 = knn.groupby('query_label',
                    {'average_neighbor_density': _gl.aggregate.AVG('density')})

        ## Join the point densities and average neighbor densities into a
        #  single SFrame and compute the local outlier factor.
        scores = scores.join(scores2, on='query_label')
        scores['anomaly_score'] = \
            scores['average_neighbor_density'] / scores['density']

        ## Remove extraneous columns and format.
        scores = scores.sort('query_label', ascending=True)
        return scores['anomaly_score']

    @classmethod
    def _get_queryable_methods(cls):
        """
        Return a list of method names that are queryable through Predictive
        Service.
        """
        return {'predict': {'dataset': 'sframe'}}
