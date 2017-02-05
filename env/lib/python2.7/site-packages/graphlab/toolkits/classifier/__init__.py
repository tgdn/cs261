"""
The GraphLab Create classifier toolkit contains models for classification
problems. Currently, we support binary classification using support vector
machines (SVM), logistic regression, boosted trees, neural networks, and nearest
neighbors. In addition to these models, we provide a smart interface that
selects the right model based on the data. If you are unsure about which model
to use, simply use :meth:`~graphlab.classifier.create` function.

Training datasets should contain a column for the 'target' variable and one or
more columns representing feature variables.

.. sourcecode:: python

    # Set up the data
    >>> import graphlab as gl
    >>> data =  gl.SFrame('https://static.turi.com/datasets/regression/houses.csv')

    # Create the model
    >>> data['is_expensive'] = data['price'] > 30000
    >>> model = gl.classifier.create(data, target='is_expensive',
    ...                              features=['bath', 'bedroom', 'size'])

    # Make predictions and evaluate results.
    >>> classification = model.classify(data)
    >>> results = model.evaluate(data)
"""

from ._classifier import create
from . import svm_classifier
from . import logistic_classifier
from . import boosted_trees_classifier
from . import neuralnet_classifier
from . import nearest_neighbor_classifier
from . import random_forest_classifier
from . import decision_tree_classifier
