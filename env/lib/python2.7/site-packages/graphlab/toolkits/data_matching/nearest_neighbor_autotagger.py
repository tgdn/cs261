import logging as _logging
import graphlab as _gl
import graphlab.connect as _mt
from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits.data_matching.autotagger import _AutoTagger, _preprocess
from graphlab.toolkits.data_matching._util import (
    distances_to_similarity_scores as _dists_to_sim_scores)
import graphlab.toolkits._internal_utils as _tkutl
from graphlab.toolkits._model_workflow import _collect_model_workflow
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._private_utils import _summarize_accessible_fields
from ._util import concat_string_features as _concat_string_features
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy


def get_default_options():
    """
    Return information about options for the nearest neighbor autotagger.

    Returns
    -------
    out : SFrame
        Each row in the output SFrames correspond to a parameter, and
        includes columns for default values, lower and upper bounds,
        description, and type.
    """
    out = _gl.SFrame({'name': ['tag_name', 'verbose']})
    out['default_value'] = ['None', 'True']
    out['description'] = [
        'Name of the column in `dataset` with the tags.',
        'Verbose printing']
    out['parameter_type'] = ['str', 'bool']

    return out

def create(dataset, tag_name=None, features=None, verbose=True):
    """
    Create a :class:`NearestNeighborAutoTagger`
    model, which can be used to quickly apply tags from a reference set of text
    labels to a new query set using the ``tag`` method.

    Parameters
    ----------
    dataset : SFrame
        Reference data. This SFrame must contain at least one column. By
        default, only the ``tag_name`` column is used as the basis for
        tagging. You may optionally include additional columns with the
        ``features`` parameter.

    tag_name : string, optional
        Name of the column in ``dataset`` with the tags. This column must
        contain string values. If ``dataset`` contains more than one column,
        ``tag_name`` must be specified.

    features : list[string], optional
        Names of the columns with features to use as the basis for tagging.
        'None' (the default) indicates that only the column specified by the
        ``tag_name`` parameter should be used. Only str or list fields are
        allowed. If a column of type list is specified, all values must be
        either of type string or convertible to type string.

    verbose : bool, optional
        If True, print verbose output during model creation.

    Returns
    -------
    out : model
        A model for quickly tagging new query observations with entries from
        `dataset`. Currently, the only implementation is the following:

        - NearestNeighborAutoTagger

    See Also
    --------
    graphlab.nearest_neighbors.NearestNeighborsModel

    Examples
    --------
    First construct a toy `SFrame` of actor names, which will serve as the
    reference set for our autotagger model.

    >>> actors_sf = gl.SFrame(
            {"actor": ["Will Smith", "Tom Hanks", "Bradley Cooper",
                       "Tom Cruise", "Jude Law", "Robert Pattinson",
                       "Matt Damon", "Brad Pitt", "Johnny Depp",
                       "Leonardo DiCaprio", "Jennifer Aniston",
                       "Jessica Alba", "Emma Stone", "Cameron Diaz",
                       "Scarlett Johansson", "Mila Kunis", "Julia Roberts",
                       "Charlize Theron", "Marion Cotillard",
                       "Angelina Jolie"]})
    >>> m = gl.data_matching.nearest_neighbor_autotagger.create(
                actors_sf, tag_name="actor")

    Then we load some IMDB movie reviews into an `SFrame` and tag them using
    the model we created above. The score field in the output is a
    similarity score, indicating the strength of the match between the query
    data and the suggested reference tag.

    >>> reviews_sf = gl.SFrame(
            "https://static.turi.com/datasets/imdb_reviews/reviews.sframe")
    >>> m.tag(reviews_sf.head(10), query_name="review", verbose=False)
    +-----------+-------------------------------+------------------+-----------------+
    | review_id |             review            |      actor       |      score      |
    +-----------+-------------------------------+------------------+-----------------+
    |     0     | Story of a man who has unn... |   Cameron Diaz   | 0.0769230769231 |
    |     0     | Story of a man who has unn... |  Angelina Jolie  | 0.0666666666667 |
    |     0     | Story of a man who has unn... | Charlize Theron  |      0.0625     |
    |     0     | Story of a man who has unn... | Robert Pattinson | 0.0588235294118 |
    |     1     | Bromwell High is a cartoon... |   Jessica Alba   |      0.125      |
    |     1     | Bromwell High is a cartoon... | Jennifer Aniston |       0.1       |
    |     1     | Bromwell High is a cartoon... | Charlize Theron  |       0.05      |
    |     1     | Bromwell High is a cartoon... | Robert Pattinson |  0.047619047619 |
    |     1     | Bromwell High is a cartoon... | Marion Cotillard |  0.047619047619 |
    |     2     | Airport '77 starts as a br... |  Julia Roberts   | 0.0961538461538 |
    |    ...    |              ...              |       ...        |       ...       |
    +-----------+-------------------------------+------------------+-----------------+

    The initial results look a little noisy. To filter out obvious spurious
    matches, we can set the `tag` method's similarity_threshold parameter.

    >>> m.tag(reviews_sf.head(1000), query_name="review", verbose=False,
              similarity_threshold=.8)
    +-----------+-------------------------------+------------------+----------------+
    | review_id |             review            |      actor       |     score      |
    +-----------+-------------------------------+------------------+----------------+
    |    341    | I caught this film at a te... |  Julia Roberts   | 0.857142857143 |
    |    657    | Fairly funny Jim Carrey ve... | Jennifer Aniston | 0.882352941176 |
    |    668    | A very funny movie. It was... | Jennifer Aniston | 0.833333333333 |
    |    673    | This film is the best film... | Jennifer Aniston |     0.9375     |
    +-----------+-------------------------------+------------------+----------------+

    In this second example, you'll notice that the ``review_id`` column is much
    more sparse. This is because all results whose score was below the specified
    similarity threshold (.8) were excluded from the output.

    """
    # validate the 'dataset' input
    _tkutl._raise_error_if_not_sframe(dataset, "dataset")
    _tkutl._raise_error_if_sframe_empty(dataset, "dataset")

    # ensure that tag_name is provided if dataset has > 1 column
    if dataset.num_cols() > 1 and not tag_name:
        raise _ToolkitError("No tag_name parameter specified on dataset " \
                            "with %d columns" % dataset.num_cols())
    tag_name = tag_name or dataset.column_names()[0]

    # ensure that column with name tag_name exists
    if tag_name not in dataset.column_names():
        raise _ToolkitError('No column named "%s" in dataset' % tag_name)

    # ensure that column is of type string
    if dataset[tag_name].dtype() != str:
        raise TypeError("The column used as the tag name must be of type " \
                        "string.")

    # use reasonable default for general case
    distance = _gl.distances.weighted_jaccard

    # if additional features are specified, ensure they are of appropriate types
    if features and not isinstance(features, list) and \
       all([isinstance(x, str) for x in features]):
        raise TypeError("The feature parameter must be a list of strings " \
                        "and those strings must correspond to columns in " \
                        "`dataset`.")

    # at a minimum, this SFrame will contain the tags as features;
    features = features or []
    features = [tag_name] + [x for x in features if x != tag_name]

    # ensure that each specified feature column is either of type list or str
    column_names = set(dataset.column_names())
    for col_name in features:
        if col_name not in column_names:
            raise _ToolkitError("Specified feature column (%s) not found " \
                                "in dataset" % col_name)

        if dataset.select_column(col_name).dtype() not in (str, list):
            raise TypeError("Only string and list columns are allowed as " \
                            "features.")

    # concatenate the feature columns into a single column
    features_sf = dataset.select_columns(features)
    feature_col, features_sf = _concat_string_features(features_sf, features)

    # compute features
    if verbose:
        _logging.getLogger().info("Extracting features...")

    features = _preprocess(features_sf.select_column(feature_col))

    # group by tag_name to ensure that tags are unique
    feature_cols = features.column_names()
    select_cols = {col_name: _gl.aggregate.SELECT_ONE(col_name) for col_name \
                   in feature_cols}
    features.add_column(dataset[tag_name], tag_name)
    features = features.groupby(tag_name, select_cols)

    # create nearest neighbors model
    m = _gl.nearest_neighbors.create(
        features, label=tag_name, distance=distance,
        features=feature_cols, verbose=verbose)

    # add standard toolkit state attributes
    state = {"nearest_neighbors_model": m,
             "training_time": m.get("training_time"),
             "tag_name": tag_name,
             "verbose": verbose,
             "num_examples": len(features),
             "features": feature_cols,
             "num_features": len(feature_cols),
             "distance": m.get("distance")}

    model = NearestNeighborAutoTagger(state)
    return model

class NearestNeighborAutoTagger(_AutoTagger, _ProxyBasedModel):
    """
    The NearestNeighborAutoTagger wraps a `NearestNeighborsModel` along with
    basic functionality for extracting features from text columns at both
    model creation and query time.

    This model should not be constructed directly. Instead, use
    :func:`graphlab.data_matching.autotagger.create` to create an instance
    of this model.
    """
    _PYTHON_NN_AUTOTAGGER_MODEL_VERSION = 2

    def __init__(self, state={}):

        if 'nearest_neighbors_model' in state:
            model = state['nearest_neighbors_model']
        else:
            model = None

        assert(isinstance(
            model, _gl.nearest_neighbors.NearestNeighborsModel))

        if model.get("distance") == "dot_product":
            raise _ToolkitError("%s is not a supported distance function for " \
                                "the NearestNeighborAutoTagger. Use %s " \
                                "instead." % ("dot_product", "cosine"))

        if model.get("distance") == "transformed_dot_product":
            raise _ToolkitError("%s is not a supported distance function for " \
                                "the NearestNeighborAutoTagger. Use %s " \
                                "instead." % ("transformed_dot_product", "cosine"))

        self.__proxy__ = _PythonProxy(state)

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
            ('Number of feature columns', 'num_features')]

        training_fields = [
            ('Total training time (seconds)', 'training_time')]

        section_titles = ['Schema', 'Training']

        return ([model_fields, training_fields], section_titles)

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """
        width = 36
        sections, section_titles = self._get_summary_struct()
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
            A description of the NearestNeighborAutoTagger.
        """
        return self.__repr__()

    def _get_version(self):
        return self._PYTHON_NN_AUTOTAGGER_MODEL_VERSION

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current NearestNeighborAutoTagger instance.

        Returns
        -------
        out : dict
            Dictionary of option and values used to train the current instance
            of the NearestNeighborAutoTagger.

        See Also
        --------
        get_default_options, list_fields, get
        """
        _mt._get_metric_tracker().track(
            self.__module__ + '.get_current_options')

        return {k: self._state[k] for k in get_default_options()['name']}

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
        An function to load a previously saved NearestNeighborAutoTagger model
        corresponding to the specified version.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handle.

        version : int
            A version number as maintained by the class writer.
        """
        _mt._get_metric_tracker().track(self.__module__ + '.load_version')

        if version < 2:
            nn_model = unpickler.load()
            state = unpickler.load()
            state['nearest_neighbors_model'] = nn_model
            return NearestNeighborAutoTagger(state)

        state = unpickler.load()
        return NearestNeighborAutoTagger(state)


    def tag(self, dataset, query_name=None, k=5, similarity_threshold=None,
            exclude_zeros=True, verbose=True):
        """
        Match the reference tags passed when a model is created to a new set of
        queries. This is a many-to-many match: each query may have any number of
        occurrences of a reference tag.

        Parameters
        ----------
        dataset : SFrame
            Query data to be tagged.

        query_name : string, optional
            Name of the column in ``dataset`` to be auto-tagged. If ``dataset``
            has more than one column, ``query_name`` must be specified.

        k : int, optional
            Number of results to return from the reference set for each query
            observation. The default is 5, but setting it to ``None`` will
            return all results whose score is greater than or equal to
            ``similarity_threshold``.

        similarity_threshold : float, optional
            Only results whose score is greater than or equal to the specified
            ``similarity_threshold`` are returned. The default is ``None``, in
            which case the ``k`` best results are returned for each query point.

        verbose : bool, optional
            If True, print progress updates and model details.

        exclude_zeros : boolean, optional
            If True, only entries for which there is a tag with a nonzero score
            are preserved in the output. This is the default behavior.

        Returns
        -------
        out : SFrame
            An SFrame with four columns:

            - row ID
            - column name specified as `tag_name` parameter to `create` method
            - column name specified as `query_name` parameter to `tag` method
            - a similarity score between 0 and 1, indicating the strength of the
              match between the query data and the suggested reference tag,
              where a score of zero indicates a poor match and a strength of 1
              corresponds to a perfect match

        Notes
        -----
        - By default, only rows for which there is a tag with a nonzero score
          are included in the output. To guarantee at least one output row for
          every input row in ``dataset``, set the ``exclude_zeros`` parameter
          to False.

        - If both ``k`` and ``similarity_threshold`` are set to ``None``, a
          ToolkitError is raised.

        Examples
        --------
        First construct a toy `SFrame` of actor names, which will serve as the
        reference set for our autotagger model.

        >>> actors_sf = gl.SFrame(
                {"actor": ["Will Smith", "Tom Hanks", "Bradley Cooper",
                           "Tom Cruise", "Jude Law", "Robert Pattinson",
                           "Matt Damon", "Brad Pitt", "Johnny Depp",
                           "Leonardo DiCaprio", "Jennifer Aniston",
                           "Jessica Alba", "Emma Stone", "Cameron Diaz",
                           "Scarlett Johansson", "Mila Kunis", "Julia Roberts",
                           "Charlize Theron", "Marion Cotillard",
                           "Angelina Jolie"]})
        >>> m = gl.data_matching.autotagger.create(actors_sf, tag_name="actor")

        Then we load some IMDB movie reviews into an `SFrame` and tag them using
        the model we created above. The score field in the output is a
        similarity score, indicating the strength of the match between the query
        data and the suggested reference tag.

        >>> reviews_sf = gl.SFrame(
                "https://static.turi.com/datasets/imdb_reviews/reviews.sframe")
        >>> m.tag(reviews_sf.head(10), query_name="review", verbose=False)
        +-----------+-------------------------------+------------------+-----------------+
        | review_id |             review            |      actor       |      score      |
        +-----------+-------------------------------+------------------+-----------------+
        |     0     | Story of a man who has unn... |   Cameron Diaz   | 0.0769230769231 |
        |     0     | Story of a man who has unn... |  Angelina Jolie  | 0.0666666666667 |
        |     0     | Story of a man who has unn... | Charlize Theron  |      0.0625     |
        |     0     | Story of a man who has unn... | Robert Pattinson | 0.0588235294118 |
        |     1     | Bromwell High is a cartoon... |   Jessica Alba   |      0.125      |
        |     1     | Bromwell High is a cartoon... | Jennifer Aniston |       0.1       |
        |     1     | Bromwell High is a cartoon... | Charlize Theron  |       0.05      |
        |     1     | Bromwell High is a cartoon... | Robert Pattinson |  0.047619047619 |
        |     1     | Bromwell High is a cartoon... | Marion Cotillard |  0.047619047619 |
        |     2     | Airport '77 starts as a br... |  Julia Roberts   | 0.0961538461538 |
        |    ...    |              ...              |       ...        |       ...       |
        +-----------+-------------------------------+------------------+-----------------+

        The initial results look a little noisy. To filter out obvious spurious
        matches, we can set the `tag` method's `similarity_threshold` parameter.

        >>> m.tag(reviews_sf.head(1000), query_name="review", verbose=False,
                  similarity_threshold=.8)
        +-----------+-------------------------------+------------------+----------------+
        | review_id |             review            |      actor       |     score      |
        +-----------+-------------------------------+------------------+----------------+
        |    341    | I caught this film at a te... |  Julia Roberts   | 0.857142857143 |
        |    657    | Fairly funny Jim Carrey ve... | Jennifer Aniston | 0.882352941176 |
        |    668    | A very funny movie. It was... | Jennifer Aniston | 0.833333333333 |
        |    673    | This film is the best film... | Jennifer Aniston |     0.9375     |
        +-----------+-------------------------------+------------------+----------------+

        """
        _mt._get_metric_tracker().track(self.__module__ + '.tag')

        # validate the 'dataset' input
        _tkutl._raise_error_if_not_sframe(dataset, "dataset")
        _tkutl._raise_error_if_sframe_empty(dataset, "dataset")

        # ensure that either k or similarity_threshold is set
        if not (k or similarity_threshold):
            raise _ToolkitError("Either k or similarity_threshold parameters " \
                                "must be set")

        # ensure that query_name is provided if dataset has > 1 column
        if dataset.num_cols() > 1 and not query_name:
            raise _ToolkitError("No query_name parameter specified on " \
                                "dataset with %d columns" % dataset.num_cols())

        query_column = query_name or dataset.column_names()[0]

        # ensure that column with name tag_name exists
        if query_column not in dataset.column_names():
            raise _ToolkitError('No column named "%s" in dataset' \
                                % query_column)

        query_sa = dataset.select_column(query_column)
        query_sf = _gl.SFrame({"id": range(len(query_sa)),
                               query_column: query_sa})

        features = _preprocess(query_sa)
        features = features.add_row_number()

        if similarity_threshold:
            if not isinstance(similarity_threshold, (float, int)):
                raise _ToolkitError("similarity_threshold parameter must be a" \
                                    "float or an int.")

            if similarity_threshold < 0 or similarity_threshold > 1:
                raise _ToolkitError("similarity_threshold parameter must be " \
                                    "between 0 and 1.")

        radius = (1 - similarity_threshold) if similarity_threshold else None

        results = self.__proxy__['nearest_neighbors_model'].query(features, label="id", k=k,
                                       radius=radius,
                                       verbose=verbose)

        # return empty SFrame immediately if no NN results
        if len(results) == 0:
            return _gl.SFrame({query_column + "_id": [],
                               query_column: [],
                               self.get("tag_name"): [],
                               "score": []})

        results = results.join(query_sf, on={"query_label": "id"})
        results.rename({"query_label": query_column + "_id"})
        results.rename({query_column: "query_label"})

        # convert distances to similarity scores
        scores = _dists_to_sim_scores("weighted_jaccard", results)

        results.add_column(scores, "score")
        results.remove_column("distance")
        results.remove_column("rank")
        results.rename({"reference_label": self.get("tag_name"),
                        "query_label": query_column})
        results.swap_columns(self.get("tag_name"), query_column)

        if exclude_zeros:
            try:
                results = results.filter_by(0.0, "score", exclude=True)
            except RuntimeError: # nothing to join
                _logging.getLogger(__name__).warn(
                    "Empty results after filtering scores of 0.")
                results = results.head(0)

        return results

    @classmethod
    def _get_queryable_methods(cls):
        """
        Returns a list of method names that are queryable from Predictive
        Services.
        """
        _mt._get_metric_tracker().track(
            cls.__module__ + '.get_queryable_methods')

        return {'tag': {'dataset':'sframe'}}

    @_collect_model_workflow
    def evaluate(self, dataset, query_name=None, k=5, similarity_threshold=None,
                 exclude_zeros=True, verbose=True):
        """
        Match the reference tags to a set of queries labeled with their true
        tags, and then evaluate the model's performance on those queries.

        The true tags should be provided as an additional column in ``dataset``,
        and that column's name should be the same as the ``tag_name`` parameter
        specified when the model was created. The type of the tags column should
        be either string or list (of strings).

        Parameters
        ----------
        dataset : SFrame
            Query data to be tagged.

        query_name : string, optional
            Name of the column in ``dataset`` to be auto-tagged. If ``dataset``
            has more than one column, ``query_name`` must be specified.

        k : int, optional
            Number of results to return from the reference set for each query
            observation. The default is 5, but setting it to ``None`` will
            return all results whose score is greater than or equal to
            ``similarity_threshold``.

        similarity_threshold : float, optional
            Only results whose score is greater than or equal to the specified
            ``similarity_threshold`` are returned. The default is ``None``, in
            which case the ``k`` best results are returned for each query point
            regardless of score.

        exclude_zeros : boolean, optional
            If True, only entries for which there is a tag with a nonzero score
            are preserved in the output. This is the default behavior.

        verbose: bool, optional
            If True, print progress updates and model details.

        Returns
        -------
        out : dict
            A dictionary containing the entire confusion matrix, as well as the
            following evaluation metrics:

            - Precision
            - Recall
            - F1 score

        See Also
        --------
        tag, graphlab.evaluation.confusion_matrix

        Notes
        -----
        - Autotagging is a variation on multiclass classification, where in
          contrast to a multiclass classifier, an autotagger model can output
          zero tags for a particular query (either because there were no tags
          with non-zero scores, or as a result of specifying a value for the
          similarity_threshold parameter). As is standard practice in multiclass
          classification, we report Precision, Recall, and F1 score as our
          evaluation metrics. Specifically, we microaverage Precision and Recall
          by counting type I errors (false positives) and type II errors (false
          negatives) over the entire confusion matrix.

        References
        ----------
        - `Wikipedia - Precision and
          recall <http://en.wikipedia.org/wiki/Precision_and_recall>`_

        - Manning, C., Raghavan P., and Schutze H. (2008). Introduction to
          Information Retrieval.

        Examples
        --------
        Continuing with the actor autotagger model referenced in previous
        example (for the ```tag``` method):

        >>> labeled_reviews_sf = gl.SFrame(
                "https://static.turi.com/datasets/imdb_reviews/reviews.10.tagged.sframe")
        >>> labeled_reviews_sf
        +-------------------------------+---------------------+
        |             review            |        actor        |
        +-------------------------------+---------------------+
        | When I saw this movie I wa... | [Leonardo DiCaprio] |
        | I rented this movie last w... |     [Matt Damon]    |
        | You've gotta hand it to St... |   [Angelina Jolie]  |
        | I caught this film at a te... |   [Julia Roberts]   |
        | I took a flyer in renting ... |  [Jennifer Aniston] |
        | Frankly I'm rather incense... |          []         |
        | This movie looked as if it... |      [Jude Law]     |
        | My wife and I watch a film... |          []         |
        | A story of amazing disinte... |          []         |
        | I don't remember a movie w... |          []         |
        +-------------------------------+---------------------+

        >>> m.evaluate(labeled_reviews_sf, query_name="review", verbose=False,
                k=1)

        .. sourcecode:: python

            {'confusion_matrix': Columns
                    count	int
                    target_label	str
                    predicted_label	str

             Rows: 10

             Data:
             +-------+-------------------+-------------------+
             | count |    target_label   |  predicted_label  |
             +-------+-------------------+-------------------+
             |   1   | Leonardo DiCaprio | Leonardo DiCaprio |
             |   1   |     Matt Damon    |     Matt Damon    |
             |   1   |   Angelina Jolie  |   Angelina Jolie  |
             |   1   |   Julia Roberts   |   Julia Roberts   |
             |   1   |  Jennifer Aniston |  Jennifer Aniston |
             |   1   |      Jude Law     |      Jude Law     |
             |   1   |        None       |     Will Smith    |
             |   1   |        None       |     Emma Stone    |
             |   1   |        None       |  Jennifer Aniston |
             |   1   |        None       |  Charlize Theron  |
             +-------+-------------------+-------------------+
             [10 rows x 3 columns],
             'f1_score': 0.7499999999999999,
             'precision': 0.6,
             'recall': 1.0}

        >>> m.evaluate(labeled_reviews_sf, query_name="review", verbose=False,
                       k=1, similarity_threshold=.6)

        .. sourcecode:: python

            {'confusion_matrix': Columns:
                    count	int
                    target_label	str
                    predicted_label	str

             Rows: 7

             Data:
             +-------+-------------------+-------------------+
             | count |    target_label   |  predicted_label  |
             +-------+-------------------+-------------------+
             |   1   | Leonardo DiCaprio | Leonardo DiCaprio |
             |   1   |   Angelina Jolie  |   Angelina Jolie  |
             |   1   |   Julia Roberts   |   Julia Roberts   |
             |   4   |        None       |        None       |
             |   1   |      Jude Law     |      Jude Law     |
             |   1   |     Matt Damon    |        None       |
             |   1   |  Jennifer Aniston |        None       |
             +-------+-------------------+-------------------+
             [7 rows x 3 columns],
             'f1_score': 0.8,
             'precision': 1.0,
             'recall': 0.6666666666666666}

        """
        _mt._get_metric_tracker().track(self.__module__ + '.tag')

        tag_name = self.get("tag_name")
        true_tags = dataset.select_column(tag_name)

        if true_tags.dtype() not in (list, str):
            raise TypeError("The %s column must either be of type str or list" % tag_name)

        if true_tags.dtype() == str:
            true_tags = true_tags.apply(lambda x: [x] if x else [])

        true_tags = true_tags.fillna([])

        dataset = dataset.select_columns([x for x in dataset.column_names() if x != tag_name])

        if similarity_threshold:
            if not isinstance(similarity_threshold, (float, int)):
                raise _ToolkitError("similarity_threshold parameter must be a" \
                                    "float or an int.")

            if similarity_threshold < 0 or similarity_threshold > 1:
                raise _ToolkitError("similarity_threshold parameter must be " \
                                    "between 0 and 1.")

        results = self.tag(dataset, query_name=query_name, k=k,
                           similarity_threshold=similarity_threshold,
                           exclude_zeros=exclude_zeros, verbose=verbose)

        if len(results) == 0:
            raise ValueError("There is no data to evaluate. Try reducing the " \
                             "similarity_threshold or increasing k.")

        group_column = (query_name or dataset.column_names()[0]) + "_id"
        dataset = dataset.add_row_number(group_column)
        results = results.groupby(group_column, {"labels": _gl.aggregate.CONCAT(tag_name)})
        results = dataset.join(results, on={group_column: group_column}, how="left")
        results = results.fillna("labels", [])
        results = results.sort(group_column)

        def precision(tps, fps):
            return tps / float(tps + fps)

        def recall(tps, fns):
            return tps / float(tps + fns)

        def f1_score(p, r):
            return 2 * p * r / (p + r)

        confusion_matrix = _gl.evaluation.confusion_matrix(true_tags, results["labels"])
        confusion_matrix = confusion_matrix.stack("target_label", "target_label")

        # TO DO: this next line will be removed once .stack type-inference is fixed
        # or type_hint parameter is exposed
        confusion_matrix = _gl.SFrame({"predicted_label": [["stub"]],
                                       "count": [1], "target_label": ["stub"]})\
                              .append(confusion_matrix)

        confusion_matrix = confusion_matrix.stack("predicted_label", "predicted_label")

        # TO DO: remove this next line, per note above
        confusion_matrix = confusion_matrix[1:]

        tps = confusion_matrix[confusion_matrix.apply(
            lambda row: row["predicted_label"] != None and \
            row["target_label"] == row["predicted_label"])]["count"].sum() or 0

        fps = confusion_matrix[confusion_matrix.apply(
            lambda row: row["predicted_label"] != None and \
            row["target_label"] != row["predicted_label"])]["count"].sum() or 0

        fns = confusion_matrix[confusion_matrix.apply(
            lambda row: row["predicted_label"] == None and \
            row["target_label"] != None)]["count"].sum() or 0

        p  = precision(tps, fps)
        r  = recall(tps, fns)
        f1 = f1_score(p, r)

        return {"precision": p, "recall": r, "f1_score": f1,
                'confusion_matrix': confusion_matrix}
