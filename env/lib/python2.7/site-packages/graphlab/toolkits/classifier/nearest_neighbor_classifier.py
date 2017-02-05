"""
Methods for creating and using a nearest neighbor classifier model.
"""

import time as _time
import copy as _copy
import array as _array
import logging as _logging
import graphlab.connect as _mt
import graphlab as _gl
from graphlab.toolkits._model import CustomModel as _CustomModel
from graphlab.toolkits._model_workflow import _collect_model_workflow
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _raise_error_if_sframe_empty
from graphlab.toolkits._internal_utils import _raise_error_if_not_sframe
from graphlab.toolkits._internal_utils import _raise_error_if_column_exists
from graphlab.toolkits._internal_utils import _raise_error_evaluation_metric_is_valid
from graphlab.toolkits._main import ToolkitError as _ToolkitError



## ---------------------------------------------- ##
## Define the default options at the module level ##
## ---------------------------------------------- ##
def get_default_options():
    """
    Return information about options for the nearest neighbor classifier.

    Returns
    -------
    out : SFrame
        Each row in the output SFrame corresponds to a parameter, and includes
        columns for default values, lower and upper bounds, description, and
        type.
    """
    out = _gl.SFrame({'name': ['distance', 'verbose']})
    out['default_value'] = [None, 'True']
    out['description'] = [
        'Function to compare dissimilarity of two records',
        'Verbose printing']
    out['lower_bound'] = [None, 0]
    out['upper_bound'] = [None, 1]
    out['parameter_type'] = ['String, function, or composite_distance',
                             'boolean']

    return out


## ---------------- ##
## Helper functions ##
## ---------------- ##
def _sort_topk_votes(x, k):
    """
    Sort a dictionary of classes and corresponding vote totals according to the
    votes, then truncate to the highest 'k' classes.
    """
    y = sorted(x.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{'class': i[0], 'votes': i[1]} for i in y]


def _construct_auto_distance(features, column_types):
    """
    Construct a composite distance function for a set of features, based on the
    types of those features.

    NOTE: This function is very similar to
    `:func:_nearest_neighbors.choose_auto_distance`. The function is separate
    because the auto-distance logic different than for each nearest
    neighbors-based toolkit.

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
    """

    ## Put input features into buckets based on type.
    numeric_ftrs = []
    string_ftrs = []
    dict_ftrs = []

    for ftr in features:
        try:
            ftr_type = column_types[ftr]
        except:
            raise ValueError("The specified feature does not exist in the " +
                             "input data.")

        if ftr_type == str:
            string_ftrs.append(ftr)

        elif ftr_type == dict:
            dict_ftrs.append(ftr)

        elif ftr_type in [int, float, _array.array]:
            numeric_ftrs.append(ftr)

        else:
            raise TypeError("Unable to automatically construct a distance " +
                            "function for feature '{}'. ".format(ftr) +
                            "For the nearest neighbor classifier, features " +
                            "must be of type integer, float, string, dictionary, " +
                            "or array.array.")

    ## Construct the distance function
    dist = []

    for ftr in string_ftrs:
        dist.append([[ftr], 'levenshtein', 1])

    if len(dict_ftrs) > 0:
        dist.append([dict_ftrs, 'weighted_jaccard', len(dict_ftrs)])

    if len(numeric_ftrs) > 0:
        dist.append([numeric_ftrs, 'euclidean', len(numeric_ftrs)])

    return dist



## -------------- ##
## Model creation ##
## -------------- ##
def create(dataset, target, features=None, distance=None, verbose=True):
    """
    Create a
    :class:`~graphlab.nearest_neighbor_classifier.NearestNeighborClassifier`
    model. This model predicts the class of a query instance by finding the most
    common class among the query's nearest neighbors.

    .. warning::

        The 'dot_product' distance is deprecated and will be removed in future
        versions of GraphLab Create. Please use 'transformed_dot_product'
        distance instead, although note that this is more than a name change; it
        is a *different* transformation of the dot product of two vectors.
        Please see the distances module documentation for more details.

    Parameters
    ----------
    dataset : SFrame
        Dataset for training the model.

    target : str
        Name of the column containing the target variable. The values in this
        column must be of string or integer type.

    features : list[str], optional
        Name of the columns with features to use in comparing records. 'None'
        (the default) indicates that all columns except the target variable
        should be used. Please note: if `distance` is specified as a composite
        distance, then that parameter controls which features are used in the
        model. Each column can be one of the following types:

        - *Numeric*: values of numeric type integer or float.

        - *Array*: array of numeric (integer or float) values. Each array
          element is treated as a separate variable in the model.

        - *Dictionary*: key-value pairs with numeric (integer or float) values.
          Each key indicates a separate variable in the model.

        - *String*: string values.

        Please note: if `distance` is specified as a composite distance, then
        that parameter controls which features are used in the model.

    distance : str, function, or list[list], optional
        Function to measure the distance between any two input data rows. This
        may be one of three types:

        - *String*: the name of a standard distance function. One of
          'euclidean', 'squared_euclidean', 'manhattan', 'levenshtein',
          'jaccard', 'weighted_jaccard', 'cosine', 'dot_product' (deprecated),
          or 'transformed_dot_product'.

        - *Function*: a function handle from the
          :mod:`~graphlab.toolkits.distances` module.

        - *Composite distance*: the weighted sum of several standard distance
          functions applied to various features. This is specified as a list of
          distance components, each of which is itself a list containing three
          items:

          1. list or tuple of feature names (str)

          2. standard distance name (str)

          3. scaling factor (int or float)

        For more information about GraphLab Create distance functions, please
        see the :py:mod:`~graphlab.toolkits.distances` module.

        For sparse vectors, missing keys are assumed to have value 0.0.

        If 'distance' is left unspecified or set to 'auto', a composite distance
        is constructed automatically based on feature types.

    verbose : bool, optional
        If True, print progress updates and model details.

    Returns
    -------
    out : NearestNeighborClassifier
        A trained model of type
        :class:`~graphlab.nearest_neighbor_classifier.NearestNeighborClassifier`.

    See Also
    --------
    NearestNeighborClassifier
    graphlab.toolkits.nearest_neighbors
    graphlab.toolkits.distances

    References
    ----------
    - `Wikipedia - nearest neighbors classifier
      <http://en.wikipedia.org/wiki/Nearest_neighbour_classifiers>`_

    - Hastie, T., Tibshirani, R., Friedman, J. (2009). `The Elements of
      Statistical Learning <http://statweb.stanford.edu/~tibs/ElemStatLearn/>`_.
      Vol. 2. New York. Springer. pp. 463-481.

    Examples
    --------
    >>> sf = graphlab.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],
    ...                       'height': [9, 25, 20, 23],
    ...                       'weight': [13, 28, 33, 22]})
    ...
    >>> model = graphlab.nearest_neighbor_classifier.create(sf, target='species')

    As with the nearest neighbors toolkit, the nearest neighbor classifier
    accepts composite distance functions.

    >>> my_dist = [[('height', 'weight'), 'euclidean', 2.7],
    ...            [('height', 'weight'), 'manhattan', 1.6]]
    ...
    >>> model = graphlab.nearest_neighbor_classifier.create(sf, target='species',
    ...                                                     distance=my_dist)
    """

    ## Set up
    ## ------
    _mt._get_metric_tracker().track('toolkit.classifier.nearest_neighbor_classifier.create')
    start_time = _time.time()


    ## Validation and preprocessing
    ## ----------------------------

    ## 'dataset' must be a non-empty SFrame
    _raise_error_if_not_sframe(dataset, "dataset")
    _raise_error_if_sframe_empty(dataset, "dataset")


    ## 'target' must be a string, in 'dataset', and the type of the target must
    #  be string or integer.
    if not isinstance(target, str) or target not in dataset.column_names():
        raise _ToolkitError("The 'target' parameter must be the name of a "
                            "column in the input dataset.")

    if not dataset[target].dtype() == str and not dataset[target].dtype() == int:
        raise TypeError("The target column must contain integers or strings.")


    ## Warn that 'None' values in the target may lead to ambiguous predictions.
    if dataset[target].num_missing() > 0:
        _logging.warning("Missing values detected in the target column. This " +
                         "may lead to ambiguous 'None' predictions, if the " +
                         "'radius' parameter is set too small in the prediction, " +
                         "classification, or evaluation methods.")


    ## convert features and distance arguments into a composite distance
    ## NOTE: this is done here instead of in the nearest neighbors toolkit
    #  because the automatic distance construction may be different for the two
    #  toolkits.
    if features is None:
        _features = [x for x in dataset.column_names() if x != target]
    else:
        _features = [x for x in features if x != target]


    if isinstance(distance, list):
        distance = _copy.deepcopy(distance)

    elif (hasattr(distance, '__call__') or
        (isinstance(distance, str) and not distance == 'auto')):
        distance = [[_features, distance, 1]]

    elif distance is None or distance == 'auto':
        col_types = {k: v for k, v in zip(dataset.column_names(),
                                          dataset.column_types())}
        distance = _construct_auto_distance(_features, col_types)

    else:
        raise TypeError("Input 'distance' not understood. The 'distance' " +
                        "parameter must be a string or a composite distance, " +
                        " or left unspecified.")


    ## Construct and query the nearest neighbors model
    ## -----------------------------------------------
    knn_model = _gl.nearest_neighbors.create(dataset, label=target,
                                             distance=distance,
                                             verbose=verbose)


    ## Postprocessing and formatting
    ## -----------------------------
    model = NearestNeighborClassifier(knn_model)
    model._state['verbose'] = verbose
    model._state['distance'] = knn_model['distance']
    model._state['num_distance_components'] = knn_model['num_distance_components']
    model._state['num_examples'] = dataset.num_rows()
    model._state['features'] = knn_model['features']
    model._state['target'] = target
    model._state['num_classes'] = len(dataset[target].unique())
    model._state['num_features'] = knn_model['num_features']
    model._state['num_unpacked_features'] = knn_model['num_unpacked_features']
    model._state['training_time'] = _time.time() - start_time
    model._target_type = dataset[target].dtype()

    return model


## ---------------- ##
## Model definition ##
## ---------------- ##
class NearestNeighborClassifier(_CustomModel):
    """
    Nearest neighbor classifier model. Nearest neighbor classifiers predict the
    class of any observation to be the most common class among the observation's
    closest neighbors.

    This model should not be constructed directly. Instead, use
    :func:`graphlab.nearest_neighbor_classifier.create` to create an instance of
    this model.
    """

    _PYTHON_NN_CLASSIFIER_MODEL_VERSION = 1

    def __init__(self, knn_model, state=None):
        if state:
            self._state = state
        else:
            self._state = {}

        assert(isinstance(knn_model, _gl.nearest_neighbors.NearestNeighborsModel))
        self._knn_model = knn_model

    def _get_version(self):
        return self._PYTHON_NN_CLASSIFIER_MODEL_VERSION

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
        _mt._get_metric_tracker().track('toolkit.classifier.nearest_neighbor_classifier.save')

        state = self._state
        pickler.dump(state)

        knn_model = self._knn_model
        pickler.dump(knn_model)

    @classmethod
    def _load_version(self, unpickler, version):
        """
        A function to load a previously saved NearestNeighborClassifier model.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handler.

        version : int
            Version number maintained by the class writer.
        """
        state = unpickler.load()
        knn_model = unpickler.load()
        if version == 0:
            knn_model._target_type = int
        return NearestNeighborClassifier(knn_model, state)

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """
        (sections, section_titles) = self._get_summary_struct()
        out = _toolkit_repr_print(self, sections, section_titles, width=36)
        return out

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : str
            A description of the NearestNeighborClassifier model.
        """
        return self.__repr__()

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
            A list of section titles.
              The order matches that of the 'sections' object.
        """
        model_fields = [
            ('Number of examples', 'num_examples'),
            ('Number of feature columns', 'num_features'),
            ('Number of unpacked features', 'num_unpacked_features'),
            ('Number of distance components', 'num_distance_components'),
            ('Number of classes', 'num_classes')]

        training_fields = [
            ('Training time (seconds)', 'training_time')]

        section_titles = [ 'Schema', 'Training Summary']
        return ([model_fields, training_fields], section_titles)

    def list_fields(self):
        """
        List the model's queryable fields.

        Returns
        -------
        out : list
            Each element in the returned list can be queried with the ``get``
            method.
        """
        return list(self._state.keys())

    def get(self, field):
        """
        Return the value contained in the model's ``field``.

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out
            Value of the requested field.

        See Also
        --------
        list_fields
        """
        try:
            return self._state[field]
        except:
            raise ValueError("There is no model field called {}".format(field))

    def __getitem__(self, field):
        return self.get(field)

    def get_current_options(self):
        """
        Return the options used to create the
        :py:class:`~.NearestNeighborClassifier` instance.

        Returns
        -------
        out : dict
            Options used to train the current instance of the
            NearestNeighborClassifier.

        See Also
        --------
        get_default_options, list_fields, get
        """
        return {k: self._state[k] for k in get_default_options()['name']}

    def classify(self, dataset, max_neighbors=10, radius=None, verbose=True):
        """
        Return the predicted class for each observation in *dataset*. This
        prediction is made based on the closest neighbors stored in the nearest
        neighbors classifier model.

        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        verbose : bool, optional
            If True, print progress updates.

        max_neighbors : int, optional
            Maximum number of neighbors to consider for each point.

        radius : float, optional
            Maximum distance from each point to a neighbor in the reference
            dataset.

        Returns
        -------
        out : SFrame
            An SFrame with model predictions. The first column is the most
            likely class according to the model, and the second column is the
            predicted probability for that class.

        See Also
        --------
        create, predict, predict_topk

        Notes
        -----
        - If the 'radius' parameter is small, it is possible that a query point
          has no qualified neighbors in the training dataset. In this case, the
          resulting class and probability for that query are 'None' in the
          SFrame output by this method. If the target column in the training
          dataset has missing values, these predictions will be ambiguous.

        - Ties between predicted classes are broken randomly.

        Examples
        --------
        >>> sf_train = graphlab.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],
        ...                             'height': [9, 25, 20, 23],
        ...                             'weight': [13, 28, 33, 22]})
        ...
        >>> sf_new = graphlab.SFrame({'height': [26, 19],
        ...                           'weight': [25, 35]})
        ...
        >>> m = graphlab.nearest_neighbor_classifier.create(sf, target='species')
        >>> ystar = m.classify(sf_new, max_neighbors=2)
        >>> print ystar
        +-------+-------------+
        | class | probability |
        +-------+-------------+
        |  dog  |     1.0     |
        | fossa |     0.5     |
        +-------+-------------+
        """
        _mt._get_metric_tracker().track(
            'toolkit.classifier.nearest_neighbor_classifier.classify')

        ## Validate the query 'dataset'. Note that the 'max_neighbors' and
        #  'radius' parameters are validated by the nearest neighbor model's
        #  query method.
        _raise_error_if_not_sframe(dataset, "dataset")
        _raise_error_if_sframe_empty(dataset, "dataset")
        n_query = dataset.num_rows()

        ## Validate neighborhood parameters 'max_neighbors'.
        # - NOTE: when the parameter name is changed in nearest neighbors, the
        #   query call will do this itself, and this block can be removed.
        if max_neighbors is not None:
            if not isinstance(max_neighbors, int):
                raise ValueError("Input 'max_neighbors' must be an integer.")

            if max_neighbors <= 0:
                raise ValueError("Input 'max_neighbors' must be larger than 0.")


        ## Find the nearest neighbors for each query and count the number of
        #  votes for each class.
        knn = self._knn_model.query(dataset, k=max_neighbors, radius=radius,
                                    verbose=verbose)

        ## If there are *no* results for *any* query make an SFrame of nothing.
        if knn.num_rows() == 0:
            ystar = _gl.SFrame(
                    {'class': _gl.SArray([None] * n_query, self._target_type),
                     'probability': _gl.SArray([None] * n_query, int)})

        else:
            ## Find the class with the most votes for each query and postprocess.
            grp = knn.groupby(['query_label', 'reference_label'], _gl.aggregate.COUNT)

            ystar = grp.groupby('query_label',
                                {'class': _gl.aggregate.ARGMAX('Count', 'reference_label'),
                                 'max_votes': _gl.aggregate.MAX('Count'),
                                 'total_votes': _gl.aggregate.SUM('Count')})

            ystar['probability'] = ystar['max_votes'] / ystar['total_votes']

            ## Fill in 'None' for query points that don't have any near neighbors.
            row_ids = _gl.SFrame({'query_label': range(n_query)})
            ystar = ystar.join(row_ids, how='right')

            ## Sort by row number (because row number is not returned) and return
            ystar = ystar.sort('query_label', ascending=True)
            ystar = ystar[['class', 'probability']]

        return ystar

    def predict(self, dataset, max_neighbors=10, radius=None,
                output_type='class', verbose=True):
        """
        Return predicted class labels for instances in *dataset*. This model
        makes predictions based on the closest neighbors stored in the nearest
        neighbors classifier model.

        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include the features used for
            model training, but does not require a target column. Additional
            columns are ignored.

        max_neighbors : int, optional
            Maximum number of neighbors to consider for each point.

        radius : float, optional
            Maximum distance from each point to a neighbor in the reference
            dataset.

        output_type : {'class', 'probability'}, optional
            Type of prediction output:

            - `class`: Predicted class label. The class with the maximum number
              of votes among the nearest neighbors in the reference dataset.

            - `probability`: Maximum number of votes for any class out of all
              nearest neighbors in the reference dataset.

        Returns
        -------
        out : SArray
            An SArray with model predictions.

        See Also
        ----------
        create, classify, predict_topk

        Notes
        -----
        - If the 'radius' parameter is small, it is possible that a query point
          has no qualified neighbors in the training dataset. In this case, the
          result for that query is 'None' in the SArray output by this method.
          If the target column in the training dataset has missing values, these
          predictions will be ambiguous.

        - Ties between predicted classes are broken randomly.

        Examples
        --------
        >>> sf_train = graphlab.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],
        ...                             'height': [9, 25, 20, 23],
        ...                             'weight': [13, 28, 33, 22]})
        ...
        >>> sf_new = graphlab.SFrame({'height': [26, 19],
        ...                           'weight': [25, 35]})
        ...
        >>> m = graphlab.nearest_neighbor_classifier.create(sf, target='species')
        >>> ystar = m.predict(sf_new, max_neighbors=2, output_type='class')
        >>> print ystar
        ['dog', 'fossa']
        """
        _mt._get_metric_tracker().track(
            'toolkit.classifier.nearest_neighbor_classifier.predict')

        ystar = self.classify(dataset=dataset, max_neighbors=max_neighbors,
                              radius=radius, verbose=verbose)

        if output_type == 'class':
            return ystar['class']

        elif output_type == 'probability':
            return ystar['probability']

        else:
            raise ValueError("Input 'output_type' not understood. 'output_type' "
                             "must be either 'class' or 'probability'.")

    def predict_topk(self, dataset,  max_neighbors=10, radius=None, k=3,
                     verbose=False):
        """
        Return top-k most likely predictions for each observation in
        ``dataset``. Predictions are returned as an SFrame with three columns:
        `row_id`, `class`, and `probability`.

        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include the features used for
            model training, but does not require a target column. Additional
            columns are ignored.

        max_neighbors : int, optional
            Maximum number of neighbors to consider for each point.

        radius : float, optional
            Maximum distance from each point to a neighbor in the reference
            dataset.

        k : int, optional
            Number of classes to return for each input example.

        Returns
        -------
        out : SFrame

        See Also
        ----------
        create, classify, predict

        Notes
        -----
        - If the 'radius' parameter is small, it is possible that a query point
          has no neighbors in the training dataset. In this case, the query is
          dropped from the SFrame output by this method. If all queries have no
          neighbors, then the result is an empty SFrame. If the target column in
          the training dataset has missing values, these predictions will be
          ambiguous.

        - Ties between predicted classes are broken randomly.

        Examples
        --------
        >>> sf_train = graphlab.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],
        ...                             'height': [9, 25, 20, 23],
        ...                             'weight': [13, 28, 33, 22]})
        ...
        >>> sf_new = graphlab.SFrame({'height': [26, 19],
        ...                           'weight': [25, 35]})
        ...
        >>> m = graphlab.nearest_neighbor_classifier.create(sf_train, target='species')
        >>> ystar = m.predict_topk(sf_new, max_neighbors=2)
        >>> print ystar
        +--------+-------+-------------+
        | row_id | class | probability |
        +--------+-------+-------------+
        |   0    |  dog  |     1.0     |
        |   1    | fossa |     0.5     |
        |   1    |  dog  |     0.5     |
        +--------+-------+-------------+
        """
        _mt._get_metric_tracker().track(
            'toolkit.classifier.nearest_neighbor_classifier.predict_topk')

        ## Validate the number of results to return. Note that the
        #  'max_neighbors' and 'radius' parameters are validated by the nearest
        #  neighbor model's query method.
        if not isinstance(k, int) or k < 1:
            raise TypeError("The number of results to return for each point, " +
                            "'k', must be an integer greater than 0.")


        ## Validate the query dataset.
        _raise_error_if_not_sframe(dataset, "dataset")
        _raise_error_if_sframe_empty(dataset, "dataset")

        ## Validate neighborhood parameters 'max_neighbors'.
        # - NOTE: when the parameter name is changed in nearest neighbors, the
        #   query call will do this itself, and this block can be removed.
        if max_neighbors is not None:
            if not isinstance(max_neighbors, int):
                raise ValueError("Input 'max_neighbors' must be an integer.")

            if max_neighbors <= 0:
                raise ValueError("Input 'max_neighbors' must be larger than 0.")


        ## Find the nearest neighbors for each query and count the number of
        #  votes for each class.
        knn = self._knn_model.query(dataset, k=max_neighbors, radius=radius,
                                    verbose=verbose)

        ## If there are *no* results for *any* query make an empty SFrame.
        if knn.num_rows() == 0:
            ystar = _gl.SFrame({'row_id': [], 'class': [], 'probability': []})
            ystar['row_id'] = ystar['row_id'].astype(int)
            ystar['class'] = ystar['class'].astype(str)


        else:
            ## Find the classes with the top-k vote totals
            grp = knn.groupby(['query_label', 'reference_label'],
                              _gl.aggregate.COUNT)

            ystar = grp.unstack(column=['reference_label', 'Count'],
                                new_column_name='votes')

            ystar['topk'] = ystar['votes'].apply(
                lambda x: _sort_topk_votes(x, k))
            ystar['total_votes'] = ystar['votes'].apply(
                lambda x: sum(x.values()))

            ## Re-stack, unpack, and rename the results
            ystar = ystar.stack('topk', new_column_name='topk')
            ystar = ystar.unpack('topk')
            ystar.rename({'topk.class': 'class', 'query_label': 'row_id'})
            ystar['probability'] = ystar['topk.votes'] / ystar['total_votes']
            ystar = ystar[['row_id', 'class', 'probability']]

        return ystar

    #@_collect_model_workflow
    def evaluate(self, dataset, metric='auto', max_neighbors=10, radius=None):
        """
        Evaluate the model's predictive accuracy. This is done by predicting the
        target class for instances in a new dataset and comparing to known
        target values.

        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include columns with the same
            names as the target and features used for model training. Additional
            columns are ignored.

        metric : str, optional
            Name of the evaluation metric.  Possible values are:

            - 'auto': Returns all available metrics.

            - 'accuracy': Classification accuracy.

            - 'confusion_matrix': An SFrame with counts of possible
              prediction/true label combinations.

            - 'roc_curve': An SFrame containing information needed for an roc
              curve (binary classification only).

        max_neighbors : int, optional
            Maximum number of neighbors to consider for each point.

        radius : float, optional
            Maximum distance from each point to a neighbor in the reference
            dataset.

        Returns
        -------
        out : dict
            Evaluation results. The dictionary keys are *accuracy* and
            *confusion_matrix* and *roc_curve* (if applicable).

        See also
        --------
        create, predict, predict_topk, classify

        Notes
        -----
        - Because the model randomly breaks ties between predicted classes, the
          results of repeated calls to `evaluate` method may differ.

        Examples
        --------
        >>> sf_train = graphlab.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],
        ...                             'height': [9, 25, 20, 23],
        ...                             'weight': [13, 28, 33, 22]})
        >>> m = graphlab.nearest_neighbor_classifier.create(sf, target='species')
        >>> ans = m.evaluate(sf_train, max_neighbors=2,
        ...                  metric='confusion_matrix')
        >>> print ans['confusion_matrix']
        +--------------+-----------------+-------+
        | target_label | predicted_label | count |
        +--------------+-----------------+-------+
        |     cat      |       dog       |   1   |
        |     dog      |       dog       |   2   |
        |    fossa     |       dog       |   1   |
        +--------------+-----------------+-------+
        """
        _mt._get_metric_tracker().track(
            'toolkit.classifier.nearest_neighbor_classifier.evaluate')

        ## Validate the metric name
        _raise_error_evaluation_metric_is_valid(metric,
                    ['auto', 'accuracy', 'confusion_matrix', 'roc_curve'])

        ## Make sure the input dataset has a target column with an appropriate
        #  type.
        target = self.get('target')
        _raise_error_if_column_exists(dataset, target, 'dataset', target)

        if not dataset[target].dtype() == str and not dataset[target].dtype() == int:
            raise TypeError("The target column of the evaluation dataset must "
                            "contain integers or strings.")

        if self._state["num_classes"] != 2:
            if (metric == 'roc_curve') or (metric == ['roc_curve']):
                err_msg  = "Currently, ROC curve is not supported for "
                err_msg += "multi-class classification in this model."
                raise _ToolkitError(err_msg)
            else:
                warn_msg  = "WARNING: Ignoring `roc_curve`. "
                warn_msg += "Not supported for multi-class classification."
                print(warn_msg)

        ## Compute predictions with the input dataset.
        ystar = self.predict(dataset, output_type='class',
                             max_neighbors=max_neighbors, radius=radius)
        ystar_prob = self.predict(dataset, output_type='probability',
                             max_neighbors=max_neighbors, radius=radius)


        ## Compile accuracy metrics
        results = {}

        if metric in ['accuracy', 'auto']:
            results['accuracy'] = _gl.evaluation.accuracy(targets=dataset[target],
                                                          predictions=ystar)

        if metric in ['confusion_matrix', 'auto']:
            results['confusion_matrix'] = \
                _gl.evaluation.confusion_matrix(targets=dataset[target],
                                                predictions=ystar)

        if self._state["num_classes"] == 2:
            if metric in ['roc_curve', 'auto']:
                results['roc_curve'] = \
                      _gl.evaluation.roc_curve(targets=dataset[target],
                                               predictions=ystar_prob)
        return results

    @classmethod
    def _get_queryable_methods(cls):
        """
        Return a list of method names that are queryable through Predictive
        Service.
        """
        return {'predict':{'dataset':'sframe'},
                'predict_topk':{'dataset':'sframe'},
                'classify':{'dataset':'sframe'}}
