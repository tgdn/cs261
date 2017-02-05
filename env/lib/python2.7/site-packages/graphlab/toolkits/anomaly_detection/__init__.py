"""
The anomaly detection toolkit identifies data points that are *different* in
some way from the rest of an input dataset. Each data point passed to a
GraphLab Create anomaly detection model is given an anomaly score from 0 to
infinity, describing how different the point is relative to the rest of the
dataset. The higher the score, the more likely a point is anomalous, according
to a given model. Often a threshold is chosen to make a final decision whether
each point is anomalous or not; this post-processing step is left to the user.

This toolkit currently includes three models, **local outlier factor** for
datasets with multiple features and independent observations, **moving
Z-score** for sequential data (typically a time series), and **bayesian changepoints**
for identifying changes in univariate data. All three of these tools
apply to *unsupervised* problems, where the user does not have training data
labeled as anonymous or typical. For cases where such training labels do exist,
please consider using the `graphlab.toolkits.classification` toolkit. If you
are unsure which model to use, the `create` function in the `anomaly_detection`
namespace will pick automatically based on the schema of your dataset.

Please see the documentation for this model for more details
about usage. The `Anomaly Detection chapter of the User Guide <https://turi.com/learn/userguide/anomaly_detection/intro.html>`_ provides
more background on anomaly detection and walks through a more in-depth example
application.
"""

from ._anomaly_detection import create
from . import local_outlier_factor
from . import moving_zscore
from . import bayesian_changepoints

