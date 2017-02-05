"""
This module provides utilities for doing text processing.

Note that standard utilities in the `text_analytics` package can be used for 
transforming text data into "bag of words" format, where a document is 
represented as a dictionary mapping unique words with the number of times that 
word occurs in the document. See :py:func:`~graphlab.text_analytics.count_words`
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

from .topic_model import perplexity
