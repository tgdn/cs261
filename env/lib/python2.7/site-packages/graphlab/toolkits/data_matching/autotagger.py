"""
Creation and abstract class methods for the autotagger models, which match text
queries to reference text tags.
"""

from graphlab.toolkits._model import CustomModel as _CustomModel
import graphlab as _gl
import graphlab.toolkits._internal_utils as _tkutl
import graphlab.connect as _mt

def _preprocess(column):
    """
    Extract basic string features: unigrams (with stopwords removed),
    bigrams, and character ngrams of length 4.

    Parameters
    ----------
    column : SArray
        A column of data from which to extract various string features.

    Returns
    -------
    out : SFrame
        An SFrame consisting of the 3 columns of data: "unigrams",
        "bigrams", and "4_shingles".

    See Also
    --------
    graphlab.text_analysis.count_words
    graphlab.text_analysis.count_ngrams
    """
    if not isinstance(column, _gl.SArray):
        raise TypeError("column parameter must be an SArray")

    features = _gl.SFrame()

    # extract unigrams, w/ stopwords filtered
    features["unigrams"] = _gl.text_analytics.count_words(column)
    features["unigrams"] = features["unigrams"].dict_trim_by_keys(
        _gl.text_analytics.stopwords(), exclude=True)

    # extract bigrams
    features["bigrams"] = _gl.text_analytics.count_ngrams(column, n=2)

    # extract character 4-grams
    features["4_shingles"] = _gl.text_analytics.count_ngrams(
        column, n=4, method="character")

    return features

def create(dataset, tag_name=None, features=None, verbose=True):
    """
    Create an autotagger model, which can be used to quickly apply tags from a
    reference set of text labels to a new query set using the `_AutoTagger.tag`
    method.

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

        - :class:`~graphlab.data_matching.autotagger.NearestNeighborAutoTagger`

    See Also
    --------
    graphlab.nearest_neighbors.NearestNeighborsModel
    graphlab.data_matching.nearest_neighbor_autotagger.NearestNeighborAutoTagger

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
    _mt._get_metric_tracker().track(__name__ + '.create')

    from . import nearest_neighbor_autotagger
    return nearest_neighbor_autotagger.create(dataset, tag_name, features, verbose)


class _AutoTagger(_CustomModel):
    """
    Abstract class for GraphLab Create AutoTagger models. This class defines
    methods common to all autotagger models but leaves unique details to
    separate model classes.
    """
    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current AutoTagger model.
        """
        raise NotImplementedError("The 'get_current_options' method has not " \
                                  "been implemented for this model.")

    def tag(self, dataset, query_name=None):
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
        """
        raise NotImplementedError(
            "_AutoTagger should not be instantiated directly. This method " \
            "is intended to be implemented in subclasses.")

    def evaluate(self, dataset):
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

        Returns
        -------
        out : dict
            A dictionary containing the following evaluation metrics:

            - Precision
            - Recall
            - F1 score
        """
        raise NotImplementedError(
            "_AutoTagger should not be instantiated directly. This method " \
            "is intended to be implemented in subclasses.")
