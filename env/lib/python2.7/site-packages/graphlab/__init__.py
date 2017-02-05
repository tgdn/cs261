"""
@package graphlab
...
GraphLab Create is a machine learning platform that enables data scientists and
app developers to easily create intelligent applications at scale. Building an
intelligent, predictive application involves iterating over multiple steps:
cleaning the data, developing features, training a model, and creating and
maintaining a predictive service. GraphLab Create does all of this in one
platform. It is easy to use, fast, and powerful.

Concise descriptions of the data structures and their methods are contained in
the API documentation, along with a small number of simple examples.

For more detailed descriptions and examples, please see the:
- `User Guide <https://turi.com/learn/userguide/>`_,
- `API Translator <https://turi.com/learn/translator/>`_,
- `How-Tos <https://turi.com/learn/how-to/>`_
- `Gallery <https://turi.com/learn/gallery/>`_.
"""

'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''

# ---------------------------------------------------------------------------
# THIS IS AN OSS OVERRIDE FILE
#
# What this means is that there is a corresponding file in the OSS directory,
# and this file overrides that. Be careful when making changes.
# Specifically, do log the differences here.
#
# - OSS does not have all the toolkits and canvas
# - OSS does not have aws
# - OSS does not have gl_numpy
# ---------------------------------------------------------------------------


from .version_info import __VERSION__, version, build_number

# early load of cython stuff. may not be needed
#import graphlab.cython.cy_server as __unused

from graphlab.util import get_runtime_config
from graphlab.util import set_runtime_config

import graphlab.connect as _mt
import graphlab.connect.aws as aws
from . import visualization

import os as _os
import sys as _sys
if _sys.platform != 'win32' or \
    (_os.path.exists(_os.path.join(_os.path.dirname(__file__), 'cython', 'libstdc++-6.dll')) and \
    _os.path.exists(_os.path.join(_os.path.dirname(__file__), 'cython', 'libgcc_s_seh-1.dll'))):
    from graphlab.data_structures.sgraph import Vertex, Edge
    from graphlab.data_structures.sgraph import SGraph
    from graphlab.data_structures.sarray import SArray
    from graphlab.data_structures.sframe import SFrame
    from graphlab.data_structures.sketch import Sketch
    from graphlab.data_structures.image import Image

    from graphlab.data_structures.sgraph import load_sgraph, load_graph

    from graphlab.toolkits._model import Model, CustomModel

    import graphlab.aggregate
    import graphlab.toolkits
    import graphlab.toolkits.clustering as clustering
    import graphlab.toolkits.distances as distances

    from graphlab.toolkits import evaluation
    import graphlab.toolkits.text_analytics as text_analytics
    import graphlab.toolkits.graph_analytics as graph_analytics
    import graphlab.toolkits._internal as _internal

    from graphlab.toolkits.graph_analytics import connected_components
    from graphlab.toolkits.graph_analytics import shortest_path
    from graphlab.toolkits.graph_analytics import kcore
    from graphlab.toolkits.graph_analytics import pagerank
    from graphlab.toolkits.graph_analytics import graph_coloring
    from graphlab.toolkits.graph_analytics import triangle_counting
    from graphlab.toolkits.graph_analytics import degree_counting
    from graphlab.toolkits.graph_analytics import label_propagation

    import graphlab.toolkits.recommender as recommender
    from graphlab.toolkits.recommender import popularity_recommender
    from graphlab.toolkits.recommender import item_similarity_recommender
    from graphlab.toolkits.recommender import ranking_factorization_recommender
    from graphlab.toolkits.recommender import item_content_recommender
    from graphlab.toolkits.recommender import factorization_recommender

    import graphlab.toolkits.data_matching as data_matching
    from graphlab.toolkits.data_matching import deduplication
    from graphlab.toolkits.data_matching import autotagger
    from graphlab.toolkits.data_matching import record_linker
    from graphlab.toolkits.data_matching import nearest_neighbor_deduplication
    from graphlab.toolkits.data_matching import nearest_neighbor_autotagger
    from graphlab.toolkits.data_matching import similarity_search

    from graphlab.toolkits.graph_analytics import connected_components
    from graphlab.toolkits.graph_analytics import shortest_path
    from graphlab.toolkits.graph_analytics import kcore
    from graphlab.toolkits.graph_analytics import pagerank
    from graphlab.toolkits.graph_analytics import graph_coloring
    from graphlab.toolkits.graph_analytics import triangle_counting
    from graphlab.toolkits.graph_analytics import degree_counting
    from graphlab.toolkits.graph_analytics import label_propagation

    import graphlab.toolkits.regression as regression
    from graphlab.toolkits.regression import boosted_trees_regression
    from graphlab.toolkits.regression import random_forest_regression
    from graphlab.toolkits.regression import decision_tree_regression
    from graphlab.toolkits.regression import linear_regression

    import graphlab.toolkits.classifier as classifier
    from graphlab.toolkits.classifier import svm_classifier
    from graphlab.toolkits.classifier import logistic_classifier
    from graphlab.toolkits.classifier import boosted_trees_classifier
    from graphlab.toolkits.classifier import random_forest_classifier
    from graphlab.toolkits.classifier import decision_tree_classifier
    from graphlab.toolkits.classifier import neuralnet_classifier
    from graphlab.toolkits.classifier import nearest_neighbor_classifier

    import graphlab.toolkits.deeplearning as deeplearning
    import graphlab.toolkits.nearest_neighbors as nearest_neighbors
    from graphlab.toolkits.clustering import kmeans
    from graphlab.toolkits.clustering import dbscan
    from graphlab.toolkits.topic_model import topic_model

    import graphlab.toolkits.anomaly_detection as anomaly_detection
    from graphlab.toolkits.anomaly_detection import local_outlier_factor
    from graphlab.toolkits.anomaly_detection import moving_zscore
    from graphlab.toolkits.anomaly_detection import bayesian_changepoints

    import graphlab.toolkits.pattern_mining as pattern_mining
    from graphlab.toolkits.pattern_mining import frequent_pattern_mining

    import graphlab.toolkits.churn_predictor as churn_predictor
    import graphlab.toolkits.lead_scoring as lead_scoring

    from graphlab.toolkits.image_analysis import image_analysis
    import graphlab.toolkits.sentiment_analysis as sentiment_analysis
    import graphlab.toolkits.product_sentiment as product_sentiment

    import graphlab.toolkits.feature_engineering as feature_engineering
    import graphlab.toolkits.model_parameter_search as model_parameter_search
    import graphlab.toolkits.cross_validation as cross_validation
    from graphlab.toolkits.model_parameter_search import grid_search
    from graphlab.toolkits.model_parameter_search import random_search
    from graphlab.toolkits.model_parameter_search import manual_search

    import graphlab.toolkits.comparison as comparison
    from graphlab.toolkits.comparison import compare
    from graphlab.toolkits.comparison import show_comparison

    import graphlab.distributed


    # internal util
    from graphlab.connect.main import launch as _launch
    from graphlab.connect.main import stop as _stop
    import graphlab.connect.main as glconnect

    ## bring load functions to the top level
    from graphlab.timeseries._timeseries import load_timeseries
    from graphlab.data_structures.sframe import load_sframe, get_spark_integration_jar_path
    from graphlab.toolkits._model import load_model
    from graphlab.data_structures.DBConnection import connect_odbc, get_libodbc_path, set_libodbc_path

    # python egg version
    __VERSION__ = '{{VERSION_STRING}}'
    version = '{{VERSION_STRING}}'

    from graphlab.version_check import get_newest_version
    from graphlab.version_check import perform_version_check
    from graphlab.util import get_environment_config
    from graphlab.util import get_graphlab_object_type
    from graphlab.util import get_log_location, get_client_log_location, get_server_log_location

    import graphlab.canvas
    from graphlab.canvas import show

    from graphlab.version_info import version
    from graphlab.version_info import __VERSION__


    class DeprecationHelper(object):
        def __init__(self, new_target):
            self.new_target = new_target

        def _warn(self):
            import warnings
            import logging
            warnings.warn("Graph has been renamed to SGraph. The Graph class will be removed in the next release.", PendingDeprecationWarning)
            logging.warning("Graph has been renamed to SGraph. The Graph class will be removed in the next release.")

        def __call__(self, *args, **kwargs):
            self._warn()
            return self.new_target(*args, **kwargs)

        def __getattr__(self, attr):
            self._warn()
            return getattr(self.new_target, attr)

    Graph = DeprecationHelper(SGraph)

    from graphlab.util import _i_am_a_lambda_worker
    from .cython import cy_pylambda_workers
    if not _i_am_a_lambda_worker():
        perform_version_check()

    ################### Extension Importing ########################
    import graphlab.extensions
    from graphlab.extensions import ext_import

    graphlab.extensions._add_meta_path()

    # rewrite the extensions module
    class _extensions_wrapper(object):
      def __init__(self, wrapped):
        self._wrapped = wrapped
        self.__doc__ = wrapped.__doc__

      def __getattr__(self, name):
        try:
            return getattr(self._wrapped, name)
        except:
            pass
        graphlab.connect.main.get_unity()
        return getattr(self._wrapped, name)

    _sys.modules["graphlab.extensions"] = _extensions_wrapper(_sys.modules["graphlab.extensions"])
    # rewrite the import
    extensions = _sys.modules["graphlab.extensions"]
    from graphlab.timeseries._timeseries import TimeSeries
    from graphlab.timeseries._grouped_timeseries import GroupedTimeSeries

    # track imports of graphlab
    _mt._get_metric_tracker().track('import.graphlab')

    # backward compatibility. Otherwise old pickles will not load
    _sys.modules["graphlab_util"] = graphlab.util
else:
    from dependencies import get_dependencies
    package_dir = _os.path.dirname(__file__)
    print("""
ACTION REQUIRED: Dependencies libstdc++-6.dll and libgcc_s_seh-1.dll not found.

1. Ensure user account has write permission to %s
2. Run graphlab.get_dependencies() to download and install them.
3. Restart Python and import graphlab again.

By running the above function, you agree to the following licenses.

* libstdc++: https://gcc.gnu.org/onlinedocs/libstdc++/manual/license.html
* xz: http://git.tukaani.org/?p=xz.git;a=blob;f=COPYING
    """ % package_dir)
