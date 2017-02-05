import graphlab as _gl
import re as _re
import os as _os
from graphlab.toolkits._model import CustomModel as _CustomModel
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy
from graphlab.toolkits._internal_utils import _toolkit_repr_print
import graphlab.connect as _mt
import operator as _operator
import graphlab as _graphlab
import logging as _logging


def create(data=None, target=None, features=None, method='auto', validation_set=None):
    """
    Create a model that trains a classifier in order to perform sentiment
    analysis on a collection of documents.

    When a target column name is not provided, a pretrained model will be used.
    The model was trained on a large collection of product review data from both
    Amazon and Yelp datasets. Predicted scores are between 0 and 1, where
    higher scores indicate more positive predicted sentiment. The model is
    a :class:`~graphlab.logistic_classifier.LogisticClassifier` model trained
    using a bag-of-words representation of
    the text data, using ratings less than 3 as negative sentiment and
    ratings of more than 3 as positive sentiment.

    Parameters
    ----------
    data: SFrame, optional
      Contains at least one column that contains the text data of interest.
      This can be unstructured text data, such as that appearing in forums,
      user-generated reviews, and so on. This is not required when using a
      pre-trained model.

    target: str, optional
      The column name containing numeric sentiment scores for each document.
      If provided, a sentiment model will be trained for the provided data set.
      If not provided, a pre-trained model will be used.

    features: list of str, optional
      The column names of interest containing text data. Each provided column
      must be str type. Defaults to using all columns of type str.

    method: str, optional
      Method to use for feature engineering and modeling. Currently only
      bag-of-words and logistic classifier ('bow-logistic') is available.

    validation_set : SFrame, optional
      A dataset for monitoring the model's generalization performance.
      This is ignored if no value is provided to the `target` argument.

    Returns
    -------
    out : :class:`~SentimentAnalysisModel`

    Examples
    --------

    You can train a sentiment analysis classifier on text data when you have
    ratings data available.

    >>> import graphlab as gl
    >>> data = gl.SFrame({'rating': [1, 5], 'text': ['hate it', 'love it']})
    >>> m = gl.sentiment_analysis.create(data, 'rating', features=['text'])
    >>> m.predict_row({'text': 'really love it'})
    >>> m.predict_row({'text': 'really hate it'})

    If you do not have ratings data, we provide a pretrained model for you to
    use as a starting point.

    >>> m = gl.sentiment_analysis.create(data, features=['text'])
    >>> m.predict(data)

    You may also evaluate predictions against known sentiment scores.

    >>> m.evaluate(data)
    """
    _mt._get_metric_tracker().track('{}.create'.format(__name__))
    logger = _logging.getLogger(__name__)

    # Validate method.
    if method == 'auto':
        method = 'bow-logistic'
    if method != 'bow-logistic':
        raise ValueError("Unsupported method provided.")

    # Check if pretrained
    if target is None:

        # Name of pretrained model: format is [name]/[version].
        model_name = 'sentiment-combined/1'

        # Download if model is not present in [tmp dir]/model_cache/.
        tmp_dir = _gl.get_runtime_config()['GRAPHLAB_CACHE_FILE_LOCATIONS']
        model_local_path = _os.path.join(tmp_dir, 'model_cache', model_name)
        model_remote_path = 'https://static.turi.com/products/graphlab-create/resources/models/python2.7/sentiment-analysis/' + model_name

        feature_extractor = _feature_extractor_for_pretrained
        if not _os.path.exists(model_local_path):
            logger.info('Downloading pretrained model...')
            m = _gl.load_model(model_remote_path)
            m.save(model_local_path)
        else:
            m = _gl.load_model(model_local_path)

        num_rows = 0

    else:
        if data is None:
            raise ValueError("The data argument is required when a target column is provided.")

        # Validate data
        # Validate features. Use all columns by default
        if features is None:
            features = data.column_names()

        # Remove target column from list of feature columns.
        features = [f for f in features if f != target]

        # Transform the target column and create the training set.
        _target = 'like'
        train = _transform_with_target(data, target, _target)

        # Process training set using the default feature extractor.
        feature_extractor = _default_feature_extractor
        train = feature_extractor(train)

        # Check for a validation set.
        kwargs = {}
        if validation_set is not None:
            validation_set = _transform_with_target(validation_set, target, _target)
            validation_set = feature_extractor(validation_set)
            kwargs['validation_set'] = validation_set

        m = _gl.logistic_classifier.create(train,
                                           target=_target,
                                           features=features,
                                           l2_penalty=.2,
                                           **kwargs)
        num_rows = data.num_rows()

    model = SentimentAnalysisModel()

    model.__proxy__.update(
        {'target':   target,
         'features': features,
         'method':   method,
         'num_rows': num_rows,
         'feature_extractor': feature_extractor,
         'classifier': m})
    return model

class SentimentAnalysisModel(_CustomModel, _ProxyBasedModel):
    _PYTHON_SENTIMENT_ANALYSIS_MODEL_VERSION = 0

    def __init__(self, state=None):
        if state is None:
            state = {}

        self.__proxy__ = _PythonProxy(state)

    def _get_version(self):
        return self._PYTHON_SENTIMENT_ANALYSIS_MODEL_VERSION

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
        _mt._get_metric_tracker().track(self.__module__ + '.load_version')
        state = unpickler.load()
        return SentimentAnalysisModel(state)


    def predict_row(self, row):
        """
        Use the model to predict sentiment of a single string.

        Parameters
        ----------
        row : dict
            A dictionary representing a single row of new observations.
            Must include keys with the same names as the features used for
            model training, but does not require a target column. Additional
            columns are ignored.


        Returns
        -------
        out : float
            Predicted sentiment, where smaller values (near 0) indicate
            negative sentiment and large values (approaching 1) indicate
            positive sentiment.

        Examples
        --------

        >>> m = gl.product_sentiment.create(sf, features=['review'])
        >>> m.predict_row({'review': "I really like this burrito."})
        """
        _mt._get_metric_tracker().track('{}.predict_row'.format(__name__))

        m = self.__proxy__['classifier']
        f = self.__proxy__['feature_extractor']
        return m.predict(f(row), output_type='probability')[0]

    def predict(self, data):
        """
        Use the model to predict sentiment of a document collection.

        Parameters
        ----------
        data : SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        Returns
        -------
        out : SArray of float
            Predicted sentiment, where smaller values (near 0) indicate
            negative sentiment and large values (approaching 1) indicate
            positive sentiment.

        Examples
        --------
        >>> import graphlab as gl
        >>> data = gl.SFrame({'rating': [1, 5], 'text': ['hate it', 'love it']})
        >>> m = gl.sentiment_analysis.create(data, 'rating', features=['text'])
        >>> m.predict(data)

        """
        _mt._get_metric_tracker().track('{}.predict'.format(__name__))

        m = self.__proxy__['classifier']
        f = self.__proxy__['feature_extractor']
        return m.predict(f(data), output_type='probability')


    def classify(self, data):
        """
        Use the model to classify sentiment of a text collection.

        Parameters
        ----------
        data : SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        Returns
        -------
        out : SArray of int
            Predicted sentiment, where 0 indicates negative sentiment and 1
            indicates positive sentiment.

        Examples
        --------
        >>> import graphlab as gl
        >>> data = gl.SFrame({'rating': [1, 5], 'text': ['hate it', 'love it']})
        >>> m = gl.sentiment_analysis.create(data, 'rating', features=['text'])
        >>> m.predict(data)

        """
        _mt._get_metric_tracker().track('{}.classify'.format(__name__))

        m = self.__proxy__['classifier']
        f = self.__proxy__['feature_extractor']
        return m.classify(f(data))

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

        data_fields = [('Number of rows', 'num_rows')]
        model_fields = [('Score column', 'target'),
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

    def evaluate(self, data, target=None, **kwargs):
        """
        Evaluate the model's predictions on held-out, labeled data.
        Metrics are computed for classifying larger-than-average sentiment.

        Parameters
        ----------

        data : SFrame
            An SFrame having the same feature columns as provided when creating
            the ProductSentiment model.

        target : str, optional
            The column name to use for sentiment scores.
            Default: Use the same column name provided when creating the
            ProductSentiment model.

        Returns
        -------
        out : dict
            The default output from a classifier.

        """
        _mt._get_metric_tracker().track('{}.evaluate'.format(__name__))

        if self.__proxy__['method'] != 'bow-logistic':
            raise NotImplementedError
        if target is None:
            if self.__proxy__['target'] is None:
                raise ValueError("No target column specified. For models that use "
                             "a pretrained model, a target column name must \n"
                             "be provided.")
            else:
                target = self.__proxy__['target']

        m = self.__proxy__['classifier']
        f = self.__proxy__['feature_extractor']
        test = _transform_with_target(data, target, 'like')
        test = f(test)
        return m.evaluate(test, **kwargs)

    def summary(self):
        """
        Get a summary for the underlying classifier.
        """
        return self.__proxy__['classifier'].summary()

    @classmethod
    def _get_queryable_methods(cls):
        '''Returns a list of method names that are queryable through Predictive
        Service'''
        return {'predict': {'data': 'sframe'},
                'predict_row': {},
                'classify': {'data': 'sframe'}}


def _get_str_columns(sf):
    """
    Returns a list of names of columns that are string type.
    """
    return [name for name in sf.column_names() if sf[name].dtype() == str]

def _default_feature_extractor(sf):
    """
    Return an SFrame containing a bag of words representation of each column.
    """
    if isinstance(sf, dict):
        out = _gl.SArray([sf]).unpack('')
    elif isinstance(sf, _gl.SFrame):
        out = sf.__copy__()
    else:
        raise ValueError("Unrecognized input to feature extractor.")
    for f in _get_str_columns(out):
        out[f] = _gl.text_analytics.count_words(out[f])
    return out

def _feature_extractor_for_pretrained(sf):
    """
    Concatenates all text columns then computes bag-of-words.
    """
    if isinstance(sf, dict):
        out = _gl.SArray([sf]).unpack('')
    elif isinstance(sf, _gl.SFrame):
        out = sf.__copy__()
    else:
        raise ValueError("Unrecognized input to feature extractor.")
    out = out[_get_str_columns(out)]
    def g(row):
        if row is None:
            return ''
        return ' '.join(row.values())
    out['bow'] = out.apply(g)
    out['bow'] = _gl.text_analytics.count_words(out['bow'])
    return out[['bow']]

def _transform_with_target(d, target, final_target_name):
    """
    For a dataset containing ratings between 1 and 5, this returns a new
    data set with a binary target column named 'like' indicating ratings of
    4 or higher. Ratings of 3 are considered ambivalent and removed.
    """
    data = d.__copy__()
    if set(data[target].unique()) != set([0, 1]):
        data = data[data[target] != 3]
        data[target] = data[target] >= 4
    data.rename({target: final_target_name})
    return data
