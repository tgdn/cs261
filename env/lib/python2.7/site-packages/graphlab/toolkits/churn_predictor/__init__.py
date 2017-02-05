"""
The GraphLab Create Churn Prediction toolkit allows predicting which users will
churn (stop using) a product or website given user activity logs. Here is a
short example of using this module on a sample dataset.

.. sourcecode:: python

    >>> import graphlab as gl
    >>> import datetime

    # Load a data set.
    >>> sf = gl.SFrame(
    ... 'https://static.turi.com/datasets/churn-prediction/online_retail.csv')

    # Convert InvoiceDate from string to datetime.
    >>> import dateutil
    >>> from dateutil import parser
    >>> sf['InvoiceDate'] = sf['InvoiceDate'].apply(parser.parse)

    # Convert SFrame into TimeSeries.
    >>> time_series = gl.TimeSeries(sf, 'InvoiceDate')

    # Create a train-test split.
    >>> train, valid = gl.churn_predictor.random_split(time_series,
    ...           user_id='CustomerID', fraction=0.9)

    # Train a churn prediction model.
    >>> model = gl.churn_predictor.create(train, user_id='CustomerID',
    ...                       features = ['Quantity'])
"""
from ._churn_predictor import create
from ._churn_predictor import ChurnPredictor
from ._churn_predictor import random_split
from graphlab.toolkits._model import _get_default_options_wrapper

get_default_options = _get_default_options_wrapper(
    '_ChurnPredictor', 'churn_predictor', 'ChurnPredictor', True)
