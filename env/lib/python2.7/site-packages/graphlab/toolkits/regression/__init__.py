r"""
The GraphLab Create regression toolkit contains models for regression problems.
Currently, we support linear regression and boosted trees. In addition to these
models, we provide a smart interface that selects the right model based on the
data. If you are unsure about which model to use, simply use
:meth:`~graphlab.regression.create` function.

Training data must contain a column for the 'target' variable and one or more
columns representing feature variables.

.. sourcecode:: python

    # Set up the data
    >>> import graphlab as gl
    >>> data =  gl.SFrame('https://static.turi.com/datasets/regression/houses.csv')

    # Select the best model based on your data.
    >>> model = gl.regression.create(data, target='price',
    ...                                  features=['bath', 'bedroom', 'size'])

    # Make predictions and evaluate results.
    >>> predictions = model.predict(data)
    >>> results = model.evaluate(data)

"""
from ._regression import create
from . import linear_regression
from . import boosted_trees_regression
from . import random_forest_regression
from . import decision_tree_regression
