"""
"""
from .platform import \
        _HTTPResponse, \
        View, \
        ViewMarkup, \
        App, \
        ViewServer, \
        get_server_instance, \
        layout, \
        wrappers

from ._sframe_base_view import SFrameBaseView
from ._overview_app import OverviewApp

from . import churn_predictor
from . import data
from . import recommender

# TODO remove the following, refactor _views.py into model-specific directory
# layout (like views/recommender, views/churn_predictor), and update callers.
ChurnPredictorDescriptionView = churn_predictor.description
ChurnPredictorExploreView = churn_predictor.explore
ChurnPredictorEvaluateView = churn_predictor.evaluate

RecommenderDescriptionView = recommender.description
RecommenderExploreView = recommender.explore
RecommenderEvaluateView = recommender.evaluate
