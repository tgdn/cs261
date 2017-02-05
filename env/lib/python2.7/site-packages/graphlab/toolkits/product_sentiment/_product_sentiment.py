import graphlab.connect as _mt
import re as _re
import graphlab as _gl
from graphlab.toolkits._model import CustomModel as _CustomModel
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy
from graphlab.toolkits._internal_utils import _toolkit_repr_print
import logging as _logging
from graphlab.deps import nltk as _nltk
from graphlab.deps import HAS_NLTK as _HAS_NLTK


def create(data, target=None, features=None, review_id=None,
           method='auto', splitby='review'):
    """
    Given user reviews of a set of products, this toolkit enables fine-grained
    analysis of review sentiments.

    For example, text from user forum could be used to better understand
    sentiment around a particular product by identifying posts that have positive
    and negative sentiment. Here each item of analysis is a forum post, and the
    user can obtain sentiment scores and summaries for particular products.

    Similarly, when analyzing review data for restaurants or movies, it is
    important to understand how sentiment varies across aspects of the reviews,
    e.g., people like the rice dishes but not the pasta dishes for a particular
    restaurant. In this case, the toolkit can analyze the set of available
    restaurant reviews and provide sentiment summaries for individual dishes of
    interest.

    The returned :py:class:`~ProductSentimentModel` enables queries of reviews
    that have positive or negative sentiment about particular keyword phrases
    via the :py:class:`~ProductSentimentModel.get_most_positive` and
    :py:class:`~ProductSentimentModel.get_most_negative` methods; the
    :py:class:`~ProductSentimentModel.sentiment_summary` method produces
    a summary of sentiment over a collection of keyword phrases.

    The user can choose to perform their analysis at one of two levels:

    - **review** : the text of each review is scored as a single snippet

    - **sentence** : each review sentence is scored separately

    Behind the scenes, each text snippet undergoes a feature engineering
    pipeline and a
    :py:class:`~graphlab.sentiment_analysis.SentimentAnalysisModel` is created
    to predict the sentiment of each snippet. The current default
    pipeline is a bag-of-words transformation of the text data, and the
    model for sentiment internally uses a
    :py:class:`~graphlab.logistic_classifier.LogisticClassifier`.
    Sometimes a target sentiment score is available (such as an associated 1-5
    stars, thumbs-up, etc.) is provided. In these instances a classifier is
    trained to predict sentiment. Predicted scores range between 0 and 1, and
    larger scores indicate a higher-than-average sentiment score. When no target
    sentiment scores exist then a pretrained model is used instead. See
    :py:class:`~graphlab.sentiment_analysis.create` for details.

    Once this :py:class:`~graphlab.sentiment_analysis.SentimentAnalysisModel` is
    created, the model can be queried for reviews (or sentences) that have
    positive or negative sentiment, as well as provide summary statistics about
    the sentiment for each keyword phrase. This is done via a
    :py:class:`~graphlab.beta.search.SearchModel` that is created on the
    text data.

    Parameters
    ----------
    data: SFrame
      Contains at least one column that contains the product review data of
      interest.

    target: str, optional
      The name of the column containing numeric sentiment scores for each review.
      If provided, a sentiment model will be trained for this data set.
      If not provided, a pre-trained model will be used.

    features: list of str, optional
      Names of columns that contain product review data. Each provided column
      must be str type. Default: uses all columns of type str.

    review_id: str, optional
      Name of the column that represents the id of each review. This value is
      returned in calls to get_most_positive and get_most_negative.
      Default: the row number is used as a review id number.

    method: str, optional
      Method to use for feature engineering and modeling. Currently only
      bag-of-words and logistic classifier ('bow-logistic') is available.

    splitby: str, optional
      Level of analysis to use for analyzing the text data.

      **review** : Each review will be considered a single snippet of text.
      Methods such as  get_most_positive will return entire reviews.

      **sentence** : Each review is first tokenized into sentences.
      Methods such as get_most_positive will return sentences. This operation
      requires the nltk package, and uses the punkt English sentence tokenizer.

    Returns
    -------
        out : :class:`~ProductSentimentModel`

    Examples
    --------

    .. sourcecode:: python

        >>> import graphlab as gl
        >>> data = gl.SFrame(
        ... 'https://static.turi.com/datasets/amazon_baby_products/amazon_baby.gl')

        >>> m = gl.product_sentiment.create(data, features=['review'])

        >>> m = gl.product_sentiment.create(data, features=['review'],
        ...                 splitby='sentence')

        >>> m.get_most_positive('diapers')['review'][0]
        'The champ  the baby bjorn  fluerville diaper bag  and graco
        pack and play bassinet all vie for the best baby purchase Great
        product  easy to use  economical  effective  absolutly fabulous
        UpdateI knew that I loved the champ  and useing the diaper genie at a
        friend s house REALLY reinforced that  '

    """
    _mt._get_metric_tracker().track('{}.create'.format(__name__))
    logger = _logging.getLogger(__name__)

    # Validate method.
    if method == 'auto':
        method = 'bow-logistic'

    # Validate features. Use all columns by default.
    if features is None:
        features = data.column_names()
    if len(features) > 1:
        raise NotImplementedError("Using more than one text column is not currently supported.")
    if isinstance(features, list):
        features = [features[0]]
    feature_column = features[0]

    # Train a model.
    sentiment = _gl.sentiment_analysis.create(data, target, features, method)

    # Store a copy of the data inside the model.
    reviews = data.__copy__()

    # If no review_id exists, add a row number.
    if review_id is None:
        review_id = '__review_id'
        reviews = reviews.add_row_number(review_id)
    review_id_column = review_id

    if splitby=='sentence':
        logger.info('Parsing into sentences...')
        reviews = _get_sentences(reviews, features[0])
        reviews = reviews.dropna()
        reviews[feature_column] = reviews['sentence']
    elif splitby != 'review':
        raise ValueError("Unexpected argument ({}) for splitby".format(splitby))

    # Determine the column name for predicted sentiment scores.
    sentiment_score_column = 'sentiment_score'
    if sentiment_score_column in reviews.column_names():
        # Try __sentiment_score instead.
        sentiment_score_column = '__sentiment_score'
        if sentiment_score_column in reviews.column_names():
            raise ValueError("Found a column named '__sentiment_score'; please "
                             "rename this column before running this method on "
                             "the data set.")

    # Score all reviews.
    reviews[sentiment_score_column] = sentiment.predict(reviews)

    # Create a SearchModel.
    search = _gl._internal.search.create(reviews, [feature_column])

    model = ProductSentimentModel()
    model.__proxy__.update({
        'target': target,
        'features': features,
        'sentiment_score_column': sentiment_score_column,
        'review_id_column': review_id_column,
        'feature_column': feature_column,
        'splitby': splitby,
        'method': method,
        'num_reviews': reviews.num_rows(),
        'sentiment_scorer': sentiment,
        'review_searcher': search,
        'reviews': reviews
    })

    return model


class ProductSentimentModel(_CustomModel, _ProxyBasedModel):
    """
    This class is useful for predicting and summarizing sentiment
    present in review data.

    This model cannot be constructed directly.  Instead, use
    :func:`graphlab.product_sentiment.create` to create an instance
    of this model.

    Examples
    --------
    .. sourcecode:: python

      >>> import graphlab as gl
      >>> data = gl.SFrame('https://static.turi.com/datasets/amazon_baby_products/amazon_baby.gl')
      >>> data = data.head(10000)

      >>> m = gl.product_sentiment.create(data, features=['review'])

    Get the 10 reviews with the highest predicted sentiment.

    >>> m.get_most_positive(k=3)

    Get the reviews with the highest predicted sentiment for these keywords
    for each unique product name.

    >>> m.get_most_positive(['stroller', 'bassinet'], groupby='name')

    Summarize the predicted sentiment for reviews that are relevant to
    the word 'toys' as well as the reviews relevant to the phrase
    'legos'.

    >>> m.sentiment_summary(['toys', 'legos'])

    When a target score column is provided, a custom model is trained.

    >>> m = gl.product_sentiment.create(data, target='rating',
                                        features=['review'])

    """

    _PRODUCT_SENTIMENT_MODEL_VERSION = 1

    def __init__(self, state=None):
        if state is None:
            state = {}

        self.__proxy__ = _PythonProxy(state)

    def _get_version(self):
        return self._PRODUCT_SENTIMENT_MODEL_VERSION

    @classmethod
    def _get_queryable_methods(cls):
        '''Returns a list of method names that are queryable through Predictive
        Service'''
        return {'get_most_positive': {},
                'get_most_negative': {},
                'sentiment_summary': {}}

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
        _mt._get_metric_tracker().track(self.__module__ + '.save_impl')
        state = self.__proxy__
        pickler.dump(state)

    @classmethod
    def _load_version(self, unpickler, version):
        """
        A function to load a previously DBSCANModel instance.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handler.

        version : int
            Version number maintained by the class writer.
        """
        state = unpickler.load()
        return ProductSentimentModel(state)

    def get_most_positive(self, keywords=None, groupby=None, k=10):
        """
        Get the snippets related to the provided keyword phrases, ranked
        by their predicted sentiment from positive to negative.

        This will use a :py:class:`~graphlab._internal.search.SearchModel` to
        obtain relevant reviews using the BM25 score for the text snippet and
        the provided keyword phrases.  See
        :func:`~graphlab._internal.search.create` for more details on that model.

        If no keywords are provided, then all reviews are included as relevant
        results.

        Parameters
        ----------
        keywords: list of str
            A list of search queries. These can be keywords or phrases that
            appear in the text columns provided as features when the model was
            created.

        groupby: str
            A column name of the original data set that should be used to group
            the results. The most positive items will be shown for each unique
            value found in this column. For instance, this could be the column
            containing product names.

        k: int
            The number of results to return for each keyword and product name.

        Returns
        -------
        out: SFrame
            Contains the original feature columns and a column of predicted
            sentiment named `sentiment_score`. When no `review_id` column name
            is provided, the results include a column of row ids named
            `__review_id`. When keyword phrases are provided, the SFrame
            includes an additional column of relevance scores. When
            `splitby=='sentence'`, then each row will correspond to a single
            sentence.

        Examples
        --------
        .. sourcecode:: python

          >>> data = gl.SFrame('https://static.turi.com/datasets/amazon_baby_products/amazon_baby.gl')
          >>> data = data.head(10000)[['name', 'review']]
          >>> m = gl.product_sentiment.create(data, features=['review'])
          >>> m.get_most_positive(k=3)
          +-------------------------------+-------------------------------+-----------------+
          |              name             |             review            | sentiment_score |
          +-------------------------------+-------------------------------+-----------------+
          | Crown Crafts The Original ... | I am the the mother of a 1... |       1.0       |
          | Chicco Hippo Hookon High C... | Our twins are 9 months old... |       1.0       |
          |  Evenflo Portable Ultrasaucer | I did a lot of research be... |       1.0       |
          +-------------------------------+-------------------------------+-----------------+

          >>> m.get_most_positive(['cheap'], groupby='name', k=3)
          +-------------+-----------------+---------+-------------------------------+
          | __review_id | relevance_score | keyword |             review            |
          +-------------+-----------------+---------+-------------------------------+
          |     3315    |  8.98588339337  |  cheap  | I am not really impressed ... |
          |     8503    |   7.742594568   |  cheap  | I purchased this item thin... |
          |     1553    |  7.55805328553  |  cheap  | The dinosaurs are very goo... |
          |     1567    |   11.078843875  |  cheap  | This is a great learning t... |
          |     1929    |  5.21646511143  |  cheap  | With 11 years between my t... |
          |     7284    |  5.10796237053  |  cheap  | I purchased this pump beca... |
          |     7235    |  10.7048436718  |  cheap  | I actually can get more mi... |
          |     7332    |  6.16265423577  |  cheap  | lol. I bought this product... |
          |     2638    |  2.00543732422  |  cheap  | My sons are 6.5 months old... |
          |     2642    |  7.13302228683  |  cheap  | My daughter already has a ... |
          +-------------+-----------------+---------+-------------------------------+
          +-------------------+--------------------------------+
          |  sentiment_score  |              name              |
          +-------------------+--------------------------------+
          |   0.987693064592  | 100% Cotton Terry Contour ...  |
          | 0.000968652689596 | 2-in-1 Car Seat Cover `n Carry |
          |   0.99992863105   | Animal Planet's Big Tub of...  |
          |   0.989658887194  | Animal Planet's Big Tub of...  |
          |   0.999956837365  | Avent ISIS Breast Pump wit...  |
          |   0.999999767378  | Avent Isis Manual Breast Pump  |
          |   0.988287661335  | Avent Isis Manual Breast Pump  |
          |   0.676748057996  | Avent Isis Manual Breast Pump  |
          |   0.999999904776  |  BABYBJORN Little Potty - Red  |
          |   0.899418593296  |  BABYBJORN Little Potty - Red  |
          +-------------------+--------------------------------+

        See Also
        --------
        create
        """
        _mt._get_metric_tracker().track('{}.get_most_positive'.format(__name__))

        return _get_most_extreme(self.__proxy__['reviews'],
                                 self.__proxy__['sentiment_scorer'],
                                 self.__proxy__['review_searcher'],
                                 self.__proxy__['review_id_column'],
                                 self.__proxy__['sentiment_score_column'],
                                 self.__proxy__['feature_column'],
                                 keywords=keywords,
                                 by=groupby, k=k, ascending=False)

    def get_most_negative(self, keywords=None, groupby=None, k=10):
        """
        Get the snippets related to the provided keyword phrases, ranked
        by their predicted sentiment from negative to positive.

        This will use a :py:class:`~graphlab._internal.search.SearchModel` to
        obtain relevant reviews using the BM25 score for the text snippet and
        the provided keyword phrases. See
        :func:`~graphlab._internal.search.create` for more details on that
        model.

        If no keywords are provided, then all reviews are included as relevant
        results.

        Parameters
        ----------
        keywords: list of str
            A list of search queries. These can be keywords or phrases that
            appear in the text columns provided as features when the model was
            created.

        groupby: str
            A column name of the original data set that should be used to group
            the results. The most positive items will be shown for each unique
            value found in this column. For instance, this could be the column
            containing product names.

        k: int
            The number of results to return for each keyword and product name.

        Returns
        -------
        out: SFrame
            Contains the original feature columns and a column of predicted
            sentiment named `sentiment_score`. When no `review_id` column name
            is provided, the results include a column of row ids named
            `__review_id`. When keyword phrases are provided, the SFrame
            includes an additional column of relevance scores. When
            `splitby=='sentence'`, then each row will correspond to a single
            sentence.

        Examples
        --------

        >>> data = gl.SFrame('https://static.turi.com/datasets/amazon_baby_products/amazon_baby.gl')
        >>> data = data.head(10000)[['name', 'review']]

        >>> m = gl.product_sentiment.create(data, features=['review'])
        >>> m.get_most_negative(k=3)
        +-------------+-------------------------------+-------------------------------+
        | __review_id |              name             |             review            |
        +-------------+-------------------------------+-------------------------------+
        |     2186    | Philips Avent 3 Pack 9oz B... | (This is a long review, bu... |
        |     7249    | Avent Isis Manual Breast Pump | As a breastfeeding Mom (fo... |
        |     4368    | Evenflo Home D&eacute;cor ... | Please do not bother with ... |
        +-------------+-------------------------------+-------------------------------+
        +-------------------+
        |  sentiment_score  |
        +-------------------+
        | 7.44384662236e-20 |
        |  3.4358691403e-12 |
        | 8.99370566357e-11 |
        +-------------------+

        >>> m.get_most_negative(['cheap'], groupby='name', k=3)
        +-------------+-----------------+---------+-------------------------------+
        | __review_id | relevance_score | keyword |             review            |
        +-------------+-----------------+---------+-------------------------------+
        |     3315    |  8.98588339337  |  cheap  | I am not really impressed ... |
        |     8503    |   7.742594568   |  cheap  | I purchased this item thin... |
        |     1567    |   11.078843875  |  cheap  | This is a great learning t... |
        |     1553    |  7.55805328553  |  cheap  | The dinosaurs are very goo... |
        |     1929    |  5.21646511143  |  cheap  | With 11 years between my t... |
        |     7332    |  6.16265423577  |  cheap  | lol. I bought this product... |
        |     7235    |  10.7048436718  |  cheap  | I actually can get more mi... |
        |     7284    |  5.10796237053  |  cheap  | I purchased this pump beca... |
        |     2642    |  7.13302228683  |  cheap  | My daughter already has a ... |
        |     2638    |  2.00543732422  |  cheap  | My sons are 6.5 months old... |
        +-------------+-----------------+---------+-------------------------------+
        +-------------------+--------------------------------+
        |  sentiment_score  |              name              |
        +-------------------+--------------------------------+
        |   0.987693064592  | 100% Cotton Terry Contour ...  |
        | 0.000968652689596 | 2-in-1 Car Seat Cover `n Carry |
        |   0.989658887194  | Animal Planet's Big Tub of...  |
        |   0.99992863105   | Animal Planet's Big Tub of...  |
        |   0.999956837365  | Avent ISIS Breast Pump wit...  |
        |   0.676748057996  | Avent Isis Manual Breast Pump  |
        |   0.988287661335  | Avent Isis Manual Breast Pump  |
        |   0.999999767378  | Avent Isis Manual Breast Pump  |
        |   0.899418593296  |  BABYBJORN Little Potty - Red  |
        |   0.999999904776  |  BABYBJORN Little Potty - Red  |
        +-------------------+--------------------------------+

        See Also
        --------
        create
        """
        _mt._get_metric_tracker().track('{}.get_most_negative'.format(__name__))

        return _get_most_extreme(self.__proxy__['reviews'],
                                 self.__proxy__['sentiment_scorer'],
                                 self.__proxy__['review_searcher'],
                                 self.__proxy__['review_id_column'],
                                 self.__proxy__['sentiment_score_column'],
                                 self.__proxy__['feature_column'],
                                 keywords=keywords,
                                 by=groupby, k=k, ascending=True)

    def sentiment_summary(self, keywords=None, groupby=None, k=10,
                          reverse=False, threshold=2):
        """
        Identify the average sentiment of each keyword phrase across all of the
        provided reviews/sentences.

        When many keyword phrases are provided, the `k` argument limits the
        results to the `k` keywords that have the highest (or lowest)
        predicted sentiment. The type of sorting is controlled by the `reverse`
        argument.

        If no keywords are provided, then all reviews are included as relevant
        results to be summarized.

        The `groupby` keyword allows for summaries  according to additional
        columns. For example, for each review there may be a column containing
        the name of the product. By providing the name of this column, you can
        get the `k` most positive/ negative keywords for each product.

        Parameters
        ----------
        keywords : list of str
          A list of search queries. Must be a subset of those provided during
          model creation.

        groupby : str
          The name of the column by which to summarize sentiment.
          For example, to summarize sentiment of several keywords for each
          unique restaurant in a dataset of restaurant reviews, this value
          should be the name of the column containing the restaurant names.

        k : int
          The number of results to show for each keyword string.

        reverse : bool
          If True, sorts with ascending sentiment scores, i.e. most negative
          sentiment to most positive sentiment. When a `groupby` parameter is
          provided, the keywords are sorted by sentiment score for each
          unique value in the column specified by `groupby`.

        threshold : int
          Limit summaries to only those having at least this number of product
          reviews.

        Returns
        -------
        out : SFrame
          The mean and standard deviation of predicted sentiment scores for
          each keyword. If no keyword is provided, then the summary statistics
          are computed across the entire set of reviews.

        Examples
        --------

        >>> import graphlab as gl
        >>> data = gl.SFrame('https://static.turi.com/datasets/amazon_baby_products/amazon_baby.gl')

        # Remove rows with empty reviews and empty product names.
        >>> data = data[data['review'] != '']
        >>> data =  data[data['name'] != '']

        >>> m = gl.product_sentiment.create(data, features=['review'])

        Get sentiment across the entire data set by not providing any keywords.

        >>> m.sentiment_summary()
        +---------------+----------------+----------------+
        | review_count  | mean_sentiment |  sd_sentiment  |
        +---------------+----------------+----------------+
        | 182384        | 0.866282635115 | 0.234768743793 |
        +---------------+----------------+----------------+

        Predict and summarize sentiment of reviews containing information
        about "lego" or "strollers".

        >>> m.sentiment_summary(['lego', 'stroller'])
        +----------+-------+----------------+----------------+
        | keyword  | count | mean_sentiment |  sd_sentiment  |
        +----------+-------+----------------+----------------+
        |   lego   |  1224 | 0.875938778376 | 0.254638934865 |
        | stroller |  9952 | 0.905175070121 | 0.211175644315 |
        +----------+-------+----------------+----------------+

        Summarize the sentiment associated with each search query by product.

        >>> m.sentiment_summary(['fantastic', 'broken'],
                                groupby='name',
                                k=3, threshold=5)
        +-----------+-------------------------------+----------------+-------------------+
        |  keyword  |              name             | mean_sentiment |    sd_sentiment   |
        +-----------+-------------------------------+----------------+-------------------+
        |   broken  | North States Industries Su... | 0.815883781164 |   0.336224065426  |
        |   broken  | Regalo My Cot Portable Bed... | 0.790486718464 |   0.139652224911  |
        |   broken  | North States Superyard Pla... | 0.745402815728 |   0.416310663621  |
        | fantastic | BABYBJORN Soft Bib 2 Pack ... | 0.999995949932 | 4.77318082569e-06 |
        | fantastic | The First Years Ignite Str... | 0.999931724352 | 6.06858810606e-05 |
        | fantastic | The First Years Jet Stroll... | 0.998533137352 |  0.00257916079049 |
        +-----------+-------------------------------+----------------+-------------------+
        +--------------+
        | review_count |
        +--------------+
        |      5       |
        |      7       |
        |      8       |
        |      6       |
        |      5       |
        |      5       |
        +--------------+

        """
        _mt._get_metric_tracker().track('{}.sentiment_summary'.format(__name__))

        return _sentsummary(self.__proxy__['reviews'],
                            self.__proxy__['sentiment_scorer'],
                            self.__proxy__['review_searcher'],
                            self.__proxy__['sentiment_score_column'],
                            self.__proxy__['feature_column'],
                            keywords,
                            groupby,
                            k,
                            reverse,
                            threshold)


    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current model.
        """
        opts = {'target'   : self.__proxy__['target'],
                'features' : self.__proxy__['features'],
                'method'   : self.__proxy__['method']
                }
        return opts

    def get_default_options(self):
        """
        Return a dictionary with the default options.
        """
        opts = {'method' : 'auto'}
        return opts

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the NearestNeighborsModel.
        """
        return self.__repr__()

    def _get_summary_struct(self):

        data_fields = [('Number of reviews', 'num_reviews'),
                       ('Split-by',      'splitby')]
        model_fields = [('Target score column', 'target'),
                        ('Features',     'features'),
                        ('Method',       'method')]

        sections = [data_fields, model_fields]
        section_titles = ['Data', 'Model']
        return sections, section_titles

    def __repr__(self):
        width = 32
        key_str = "{:<{}}: {}"
        (sections, section_titles) = self._get_summary_struct()
        out = _toolkit_repr_print(self, sections, section_titles, width=width)
        return out


def _split(sf, cols):
    """
    Partition an SFrame in a set of SFrames, one for each unique
    combination of values in the chosen columns.

    Parameters
    ----------
    sf : SFrame

    cols : list of str

    Returns
    -------
    out : list of pairs
      Each pair has a first element that is one of the unique values,
      and the second element is the partition of the SFrame that has
      that unique value.
    """
    key_col = sf.apply(lambda x: '_'.join([x[c] for c in cols]))
    sfs = [(u, sf[key_col == u]) for u in key_col.unique().sort()]
    return sfs

def _apply(sfs, f):
    """
    Apply the provided function to each SFrame in the list as produced
    by _combine.
    """
    return [(k, f(v)) for k, v in sfs]

def _combine(sfs):
    """
    Combine a list of SFrames by appending them.
    """
    if len(sfs) == 0:
        return _gl.SFrame()
    sf = sfs[0][1]
    if len(sfs) == 1:
        return sf
    for (k, v) in sfs[1:]:
        sf = sf.append(v)
    return sf

def _groupby_sort(sf, groupby_keys, sort_keys, ascending=False):
    """
    Helper function that
    - splits on distinct rows of sf[groupby_keys],
    - sorts each chunk according to sort_keys,
    - then returns the appended SFrame.
    """
    sfs = _split(sf, groupby_keys)
    def f(sf):
        return sf.sort(sort_keys, ascending=ascending)
    sfs = _apply(sfs, f)
    return _combine(sfs)

def _groupby_topk(sf, group, key, value, extra_cols=None, k=10, ascending=False):
    """
    Helper function that
    - splits on distinct rows of sf[[group]],
    - runs topk on each chunk according to the column named value using the
      provided k value.
    - returns only the [group, key, value] columns of the appended SFrame.
    """

    sfs = _split(sf, [group])
    def f(sf):
        return sf.topk(value, k=k, reverse=ascending)
    sfs = _apply(sfs, f)
    sf = _combine(sfs)
    cols = [group, key, value]
    if extra_cols is not None:
        cols += extra_cols
    sf = sf[cols]
    return sf

def _txt2sentences(txt, remove_none_english_chars=True):
    """
    Split the English text into sentences using NLTK
    :param txt: input text.
    :param remove_none_english_chars: if True then remove non-english chars from text
    :return: string in which each line consists of single sentence from the original input text.
    :rtype: str
    """

    if not _HAS_NLTK:
        raise ValueError("NLTK currently required for sentence tokenization, "
                         "but no NLTK installation was detected")
    _nltk.download("punkt", quiet=True)
    tokenizer = _nltk.data.load('tokenizers/punkt/english.pickle')

    # split text into sentences using nltk packages
    for s in tokenizer.tokenize(txt):
        if remove_none_english_chars:
            #remove none English chars
            s = _re.sub("[^a-zA-Z]", " ", s)
        yield s

def _get_sentences(data, column_name):
    """
    Helper function to create a version of the data set where there is a row
    for each sentence of the review.
    """
    d = data.__copy__()
    def f(x):
        x = x.encode().decode('ascii', 'ignore')
        return list(_txt2sentences(x))
    d['sentence'] = d[column_name].apply(f)
    d = d.stack('sentence', 'sentence')
    return d

def _get_most_extreme(reviews, scorer, searcher,
                      review_id_column,
                      sentiment_score_column,
                      feature_column,
                      keywords=None,
                      by=None,
                      k=10,
                      ascending=False):
    """
    Helper function for get_most_positive and get_most_negative.
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    how = feature_column
    keyword_name = 'keyword'

    if not isinstance(k, int) or k < 0:
        raise ValueError("The `k` argument must be a non-negative integer.")

    if how == by:
        raise ValueError('The provided `groupby` parameter represents the same '
                         'column name that was used by the model for the text '
                         'of the review. Please choose a different column to '
                         'groupby.')

    # Determine which columns will be shown.
    cols = [how, sentiment_score_column]
    if keywords is not None:
        cols = [keyword_name] + cols
    if by is not None:
        cols = cols + [by]
    extra_cols = [review_id_column, 'relevance_score']

    if keywords is None:
        if by is not None:
            r = _groupby_topk(reviews, by, how, sentiment_score_column,
                              extra_cols=[review_id_column],
                              k=k, ascending=ascending)
        else:
            r = reviews.topk(sentiment_score_column, k=k, reverse=ascending)
        return r

    # For each keyword, get the subset of items ranked by sentiment.
    ranked = []
    for keyword in keywords:
        subsf = searcher.query(keyword)
        subsf.rename({'score': 'relevance_score'})

        if subsf.num_rows() == 0:
            continue

        if by is not None:
            r = _groupby_topk(subsf, by, how, sentiment_score_column,
                              extra_cols=extra_cols,
                              k=k, ascending=ascending)
        else:
            r = subsf.topk(sentiment_score_column, k=k, reverse=ascending)

        # Append to list of results
        r[keyword_name] = keyword
        ranked.append((keyword, r))

    # Combine ranked results together
    s = _combine(ranked)

    if s.num_rows() == 0:
        return gl.SFrame()
    return s[extra_cols + cols]

def _sentsummary(reviews, scorer, searcher,
                 sentiment_score_column,
                 feature_column,
                 keywords=None, by=None, k=5,
                 ascending=False, threshold=2):
    """
    Helper function that
    - searches for each keyword. When there are no relevant search results, an
      empty SFrame is returned.
    - performs a groupby on that keyword and optionally another column
    - thresholds any places that do not appear above a given threshold
    - sorts and shows only the top (or bottom) k.

    Parameters
    ----------
    reviews : SFrame
        The original data, which contains the column named feature_column
        and a score column containing sentiment estimates.

    scorer : SentimentAnalysisModel
        A trained model.

    searcher : SearchModel
        A trained SearchModel.

    feature_column : str
        The name of the column containing text.

    keywords : list of str
        The search keywords by which to summarize sentiment.

    by : str
        The column to groupby.

    k : int
        The number of results to show.

    Returns
    -------
    out : SFrame

    """
    if not isinstance(k, int) or k < 0:
        raise ValueError("The `k` argument must be a non-negative integer.")

    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError("The `threshold` argument must be a non-negative integer.")

    if isinstance(keywords, str):
        keywords = [keywords]

    how = feature_column
    if how == by:
        raise ValueError('The provided `groupby` parameter represents the same '
                         'column name that was used by the model for the text '
                         'of the review. Please choose a different column to '
                         'groupby.')


    aggs = {'review_count':          _gl.aggregate.COUNT(),
            'mean_sentiment': _gl.aggregate.MEAN(sentiment_score_column),
            'sd_sentiment':   _gl.aggregate.STD(sentiment_score_column)}

    keyword_name = 'keyword'

    # Decide groupby key columns.
    if by is None:
        groupby = [keyword_name]
    elif isinstance(by, str):
        groupby = [keyword_name, by]
    elif isinstance(by, list):
        groupby = [keyword_name] + by
    else:
        raise ValueError("Unexpected type for `by`. Must be str or list of str.")

    cols = [how, sentiment_score_column] + groupby

    # Aggregate entire dataset when keywords are not provided.
    if keywords is None:
        reviews[keyword_name] = '__keyword'
        result = reviews.groupby(groupby, aggs)
        del reviews[keyword_name]
        del result[keyword_name]
        result = result[result['review_count'] >= threshold]
        result = result.topk('mean_sentiment', k=k, reverse=ascending)
        return result

    # Search for each keyword
    keyword_results= []
    for keyword in keywords:
        search_results = searcher.query(keyword, num_results=k)
        search_results.rename({'score': 'relevance_score'})
        search_results[keyword_name] = keyword
        if search_results.num_rows() > 0:
            keyword_results.append((keyword, search_results))

    # Combine results
    result = _combine(keyword_results)

    # If no relevant results, return an empty SFrame.
    if result.num_rows() == 0:
        return _gl.SFrame()

    # Summarize sentiment.
    result = result.groupby(groupby, aggs)

    # Filter out rows that have too few relevant results.
    result = result[result['review_count'] >= threshold]

    # If no relevant results, return an empty SFrame.
    if result.num_rows() == 0:
        return _gl.SFrame()

    # If not using groupby, just sort.
    if by is None:
        result = result.topk('mean_sentiment', k=k, reverse=ascending)
    else:
        result = _groupby_topk(result, keyword_name, by, 'mean_sentiment',
                               extra_cols=['sd_sentiment', 'review_count'],
                               k=k, ascending=ascending)

    return result
