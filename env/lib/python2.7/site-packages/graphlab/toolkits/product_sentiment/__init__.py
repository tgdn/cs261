"""
This package contains methods for predicting and summarizing sentiment
present in review data.
"""
from ._product_sentiment import create
from ._product_sentiment  import ProductSentimentModel
from ._product_sentiment import _split, _apply, _combine, _groupby_sort, _groupby_topk, _txt2sentences, _get_sentences, _get_most_extreme, _sentsummary
