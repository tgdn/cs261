"""
This module provides utilities for doing text processing.

Note that standard SArray utilities can be used for transforming text data
into "bag of words" format, where a document is represented as a
dictionary mapping unique words with the number of times that word occurs
in the document. See :py:func:`~graphlab.text_analytics.count_words`
for more details. Also, see :py:func:`~graphlab.SFrame.pack_columns` and
:py:func:`~graphlab.SFrame.unstack` for ways of creating SArrays
containing dictionary types.

We provide methods for learning topic models, which can be useful for modeling
large document collections. See
:py:func:`~graphlab.topic_model.create` for more, as well as the
`How-Tos <https://turi.com/learn/how-to>`_, data science `Gallery
<https://turi.com/learn/gallery>`_, and `text analysis chapter of
the User Guide
<https://turi.com/learn/userguide/text/intro.html>`_.

"""

__all__ = ['tf_idf', 'bm25', 'stopwords', 'count_words',
           'count_ngrams', 'random_split', 'parse_sparse',
           'parse_docword', 'tokenize', 'trim_rare_words','split_by_sentence',
           'extract_parts_of_speech']
def __dir__():
  return ['tf_idf', 'bm25', 'stopwords', 'count_words',
          'count_ngrams', 'random_split', 'parse_sparse',
          'parse_docword', 'tokenize', 'trim_rare_words', 'split_by_sentence',
          'extract_parts_of_speech']
from ._util import tf_idf
from ._util import bm25
from ._util import stopwords
from ._util import count_words
from ._util import count_ngrams
from ._util import random_split
from ._util import parse_sparse
from ._util import parse_docword
from ._util import tokenize
from ._util import trim_rare_words
from ._util import split_by_sentence
from ._util import extract_parts_of_speech

from ._parts_of_speech import PartOfSpeech
