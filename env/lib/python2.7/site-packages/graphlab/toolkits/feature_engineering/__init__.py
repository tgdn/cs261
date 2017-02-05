'''
A transformer is a stateful object that transforms input data (as an SFrame)
from one form to another. Transformers are commonly used for feature
engineering. In addition to the modules provided in GraphLab create, users can
write transformers that integrate seamlessly with already existing ones.


Each transformer has the following methods:

    +---------------+---------------------------------------------------+
    |   Method      | Description                                       |
    +===============+===================================================+
    | __init__      | Construct the object.                             |
    +---------------+---------------------------------------------------+
    | fit           | Fit the object using training data.               |
    +---------------+---------------------------------------------------+
    | transform     | Transform the object on training/test data.       |
    +---------------+---------------------------------------------------+
    | fit_transform | First perform fit() and then transform() on data. |
    +---------------+---------------------------------------------------+
    | save          | Save the model to a GraphLab Create archive.      |
    +---------------+---------------------------------------------------+
'''
from ._feature_engineering import Transformer as _Transformer
from ._feature_engineering import _SampleTransformer

from ._feature_engineering import TransformerBase
from ._transformer_chain import TransformerChain
from ._feature_hasher import FeatureHasher
from ._dimension_reduction import RandomProjection
from ._quadratic_features import QuadraticFeatures
from ._one_hot_encoder import OneHotEncoder
from ._count_thresholder import CountThresholder
from ._feature_binner import FeatureBinner
from ._numeric_imputer import NumericImputer
from ._tokenizer import Tokenizer
from ._tfidf import TFIDF
from ._bm25 import BM25
from ._word_counter import WordCounter
from ._ngram_counter import NGramCounter
from ._word_trimmer import RareWordTrimmer
from ._categorical_imputer import CategoricalImputer
from ._deep_feature_extractor import DeepFeatureExtractor
from ._count_featurizer import CountFeaturizer
from ._sentence_splitter import SentenceSplitter
from ._part_of_speech_extractor import PartOfSpeechExtractor
from ._transform_to_flat_dictionary import TransformToFlatDictionary
from ._autovectorizer import AutoVectorizer
from graphlab.toolkits._internal_utils import _raise_error_if_not_sframe

def create(dataset, transformers):
    """
    Create a Transformer object to transform data for feature engineering.

    Parameters
    ----------
    dataset : SFrame
        The dataset to use for training the model.

    transformers: Transformer  | list[Transformer]
        An Transformer or a list of Transformers.

    See Also
    --------
    graphlab.toolkits.feature_engineering._feature_engineering._TransformerBase

    Examples
    --------

    .. sourcecode:: python

        # Create data.
        >>> sf = graphlab.SFrame({'a': [1,2,3], 'b' : [2,3,4]})

        >>> from graphlab.toolkits.feature_engineering import FeatureHasher, \
                                               QuadraticFeatures, OneHotEncoder

        # Create a single transformer.
        >>> encoder = graphlab.feature_engineering.create(sf,
                                 OneHotEncoder(max_categories = 10))

        # Create a chain of transformers.
        >>> chain = graphlab.feature_engineering.create(sf, [
                                    QuadraticFeatures(),
                                    FeatureHasher()
                                  ])

        # Create a chain of transformers with names for each of the steps.
        >>> chain = graphlab.feature_engineering.create(sf, [
                                    ('quadratic', QuadraticFeatures()),
                                    ('hasher', FeatureHasher())
                                  ])


    """
    err_msg = "The parameters 'transformers' must be a valid Transformer object."
    cls = transformers.__class__

    _raise_error_if_not_sframe(dataset, "dataset")

    # List of transformers.
    if (cls == list):
        transformers = TransformerChain(transformers)
    # Transformer.
    else:
        if not issubclass(cls, TransformerBase):
            raise TypeError(err_msg)
    # Fit and return
    transformers.fit(dataset)
    return transformers
