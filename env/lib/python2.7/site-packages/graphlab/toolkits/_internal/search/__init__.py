"""
This module provides utilities for making a set of text documents searchable.

"""

__all__ = ['search']
def __dir__():
    return ['search']
from ._search import create
from ._search import SearchModel
