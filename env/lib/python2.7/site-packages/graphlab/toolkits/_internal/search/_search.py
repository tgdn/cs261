import sys as _sys

import graphlab as _gl
import graphlab.connect.main as glconnect
from graphlab.toolkits._internal_utils import _raise_error_if_not_sframe
from graphlab.toolkits._model import SDKModel as _SDKModel
from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.util import _make_internal_url
from graphlab.util import _raise_error_if_not_of_type
from graphlab.util import _raise_error_if_not_of_type

def create(data, features=None,
           bm25_k1=1.5,
           bm25_b=0.75,
           tfidf_threshold=0.01,
           verbose=True):
    """
    Create a searchable index of text columns in an SFrame.

    Parameters
    ----------
    data : SFrame
      An SFrame containing at least one str column containing text that should
      be indexed.

    features : list of str
      A list of column names that contain text that should be indexed.
      Default: all str columns in the provided dataset.

    bm25_k1 : float
      Tuning parameter for the relative importance of term frequencies when
      computing the BM25 score between a query token and a document.

    bm25_b : float
      Tuning parameter to downweight scores of long documents when
      computing the BM25 score between a query token and a document.

    tfidf_threshold : float
      Tuning parameter to skip indexing words that have a TF-IDF score below
      this value.

    verbose : bool
      Controls whether or not to print progress during model creation.

    Returns
    -------
    out
       SearchModel

    See Also
    --------
    SearchModel.query

    References
    ----------

    Christopher D. Manning, Hinrich Schutze, and Prabhakar Raghavan.
    Introduction to information retrieval.
    http://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf

    Examples
    --------

    >>> import graphlab as gl
    >>> sf = gl.SFrame({'text': ['Hello my friend', 'I love this burrito']})
    >>> m = gl.toolkits._internal.search.create(sf)
    >>> print m.query('burrito')

    """

    # Input validation on data and features
    if features is None:
        features = _get_str_columns(data)

    _raise_error_if_not_of_type(data, [_gl.SFrame])
    _raise_error_if_not_of_type(features, [list])
    for f in features:
        if data[f].dtype() != str:
            raise _ToolkitError("Feature `%s` must be of type str" % f)

    # Store options
    options = {}
    options['bm25_b'] = bm25_b
    options['bm25_k1'] = bm25_k1
    options['tfidf_threshold'] = tfidf_threshold
    options['verbose'] = verbose
    options['features'] = features

    # Construct model
    proxy = _gl.extensions._SearchIndex()
    proxy.init_options(options)
    proxy.index(data)

    return SearchModel(proxy)

class SearchModel(_SDKModel):
    """
    SearchModel objects can be used to search text data for a given query.

    This model should not be constructed directly. Instead, use
    :func:`graphlab.toolkits._internal.search.create` to create an
    instance of this model.
    """

    def __init__(self, model_proxy=None):
        super(SearchModel, self).__init__(model_proxy)
        self.__name__ = 'search'

    def _get_wrapper(self):
        _class = self.__proxy__.__class__
        proxy_wrapper = self.__proxy__._get_wrapper()
        def model_wrapper(unity_proxy):
            model_proxy = proxy_wrapper(unity_proxy)
            return SearchModel(model_proxy)
        return model_wrapper

    @classmethod
    def _get_queryable_methods(cls):
        '''Returns a list of method names that are queryable through Predictive
        Service'''
        return {'query': {}}

    def get_current_options(self):
        return self.__proxy__.get_current_options()

    def __str__(self):
        return self.__repr__()

    def _get_summary_struct(self):
        """
        Returns a structured description of the model, including (where relevant)
        the schema of the training data, description of the training data,
        training statistics, etc.

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
        data_fields = [
            ('Number of documents',     'num_documents'),
            ('Average tokens/document', 'average_document_length')]
        param_ranking_fields = [
            ('BM25 k1',                 'bm25_k1'),
            ('BM25 b',                  'bm25_b'),
            ('TF-IDF threshold',        'tfidf_threshold')]
        index_fields = [
            ('Number of unique tokens indexed', 'num_tokens'),
            ('Preprocessing time (s)',  'elapsed_processing'),
            ('Indexing time (s)',       'elapsed_indexing')]
        section_titles = ['Corpus',
                          'Indexing settings',
                          'Index']
        return ([data_fields,
                 param_ranking_fields,
                 index_fields],
                section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections,
                                         section_titles, width=32)


    def query(self, query, num_results=10,
              expansion_k=5,
              expansion_epsilon=0.1,
              expansion_near_match_weight=.5):
        """
        Search for text.

        Parameters
        ----------
        query: str
            A string of text.

        num_results : int
            The number of results to return.

        expansion_k : int
          Maximum number of nearest words to include from query token.

        expansion_epsilon : float
          Maximum distance to allow between query token and nearby word when
          doing query expansion. Must be between 0 and 1.

        expansion_near_match_weight : float
          Multiplier to use on BM25 scores for documents indexed via an
          approximate match with a given token. This will be used for each of
          the `expansion_k` words that are considered an approximate match.
          Must be between 0 and 1.

        Returns
        -------
        out: SFrame
          The rows of the original SFrame along with a `score` column
          which contains the BM25 score between this query and the row.

        Examples
        --------

        >>> import graphlab as gl
        >>> sf = gl.SFrame({'text': ['Hello my friend', 'I love this burrito']})
        >>> s = gl.search.create(sf, features=['text'])
        >>> s.query('burrito')

        """
        if _sys.version_info.major == 2:
            _raise_error_if_not_of_type(query, [str, unicode])
        else:
            _raise_error_if_not_of_type(query, [str])
        q = query.split(' ')
        results = self.__proxy__.query_index(q,
                        expansion_k=expansion_k,
                        expansion_epsilon=expansion_epsilon,
                        expansion_near_match_weight=expansion_near_match_weight)
        results = self.__proxy__.join_query_result(results, method='default',
                                                num_results=num_results)

        return results


def _get_str_columns(sf):
    """
    Returns a list of names of columns that are string type.
    """
    return [name for name in sf.column_names() if sf[name].dtype() == str]
