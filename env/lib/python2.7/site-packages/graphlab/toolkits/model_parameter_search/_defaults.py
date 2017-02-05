from collections import OrderedDict as _OrderedDict
import graphlab as _gl
from graphlab.deps import HAS_SKLEARN


def _sensible_defaults(model_factory, data=None):
    _defaults = {
    _gl.linear_regression.create: _OrderedDict(
      [('l2_penalty', [0.0, 0.0001, 0.001, 0.01, .1, 1.0, 10.0, 100.0]),
       ('l1_penalty', [0.0, 0.0001, 0.001, 0.01, .1, 1.0, 10.0, 100.0])]),
    _gl.ranking_factorization_recommender.create: _OrderedDict(
      [('num_factors', [8, 16, 32, 64]),
       ('max_iterations', [25, 50]),
       ('regularization', [1e-9, 1e-8, 1e-7, 1e-6, 1e-4]),
       ('num_sampled_negative_examples', [4, 8]),
       ('ranking_regularization', [.1, .25, .5])]),
    _gl.factorization_recommender.create: _OrderedDict(
      [('num_factors', [8, 16, 32, 64]),
       ('max_iterations', [25, 50]),
       ('regularization', [1e-9, 1e-8, 1e-7, 1e-6, 1e-4]),
       ('linear_regularization', [1e-9, 1e-7, 1e-5])]),
    _gl.boosted_trees_classifier.create: _OrderedDict(
      [('early_stopping_rounds', [5]),
       ('max_depth', [4, 6, 8, 10]),
       ('max_iterations', [100]),
       ('min_loss_reduction', [0, 1, 10]),
       ('step_size', [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5]),
       ('column_subsample', [1, .9, .8]),
       ('row_subsample', [1, .9]),
       ('min_child_weight', [1, 2, 4, 8, 16])]),
    _gl.boosted_trees_regression.create: _OrderedDict(
      [('early_stopping_rounds', [5]),
       ('max_depth', [4, 6, 8, 10]),
       ('max_iterations', [100]),
       ('min_loss_reduction', [0, 1, 10]),
       ('step_size', [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5]),
       ('column_subsample', [1, .9, .8]),
       ('row_subsample', [1, .9]),
       ('min_child_weight', [1, 2, 4, 8, 16])]),
    _gl.decision_tree_classifier.create: _OrderedDict(
      [('max_depth', [4, 6, 8, 10]),
       ('min_loss_reduction', [0, 1, 10]),
       ('min_child_weight', [1, 2, 4, 8, 16])]),
    _gl.decision_tree_regression.create: _OrderedDict(
      [('max_depth', [4, 6, 8, 10]),
       ('min_loss_reduction', [0, 1, 10]),
       ('min_child_weight', [1, 2, 4, 8, 16])]),
    _gl.random_forest_classifier.create: _OrderedDict(
      [('max_depth', [4, 6, 8, 10]),
       ('max_iterations', [100]),
       ('min_loss_reduction', [0, 1, 10]),
       ('column_subsample', [1, .9, .8]),
       ('row_subsample', [1, .9]),
       ('min_child_weight', [1, 2, 4, 8, 16])]),
    _gl.random_forest_regression.create: _OrderedDict(
      [('max_depth', [4, 6, 8, 10]),
       ('max_iterations', [100]),
       ('min_loss_reduction', [0, 1, 10]),
       ('column_subsample', [1, .9, .8]),
       ('row_subsample', [1, .9]),
       ('min_child_weight', [1, 2, 4, 8, 16])]),
    _gl.logistic_classifier.create: _OrderedDict(
      [('l2_penalty', [0.0, 0.0001, 0.001, 0.01, .1, 1.0, 10.0, 100.0]),
       ('l1_penalty', [0.0, 0.0001, 0.001, 0.01, .1, 1.0, 10.0, 100.0])]),
    _gl.neuralnet_classifier.create: _OrderedDict(
      [('learning_rate', [0.0001, 0.001, 0.01])]),
    _gl.svm_classifier.create: _OrderedDict(
      [('penalty', [0.001, 0.01, 0.1, 1.0, 10.0])]),
    _gl.topic_model.create: _OrderedDict(
      [('num_topics', [10, 20, 50])]),
    _gl.kmeans.create: _OrderedDict(
      [('num_clusters', range(2, 21))])
      }

    if HAS_SKLEARN:
        from sklearn.svm import SVC, LinearSVC
        from sklearn.linear_model import ElasticNet, LogisticRegression, LinearRegression
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier

        _defaults[LogisticRegression] = _OrderedDict(
          [('penalty', ['l1', 'l2']),
           ('C',       [0.01, 0.1, 1.0, 2.0, 3.0, 10.0])])
        _defaults[SVC] = _OrderedDict(
          [('C',       [0.01, 0.1, 1.0, 2.0, 3.0, 10.0]),
           ('kernel',  ['rbf', 'linear', 'poly', 'rbf', 'sigmoid']),
           ('degree',  [2, 3]),
           ('probability', [True, False])])
        _defaults[LinearSVC] = _OrderedDict(
          [('C',       [0.01, 0.1, 1.0, 2.0, 3.0, 10.0]),
           ('loss',    ['squared_hinge', 'hing']),
           ('penalty', ['l2', 'l1']),
           ('dual',    [True, False])])
        _defaults[ElasticNet] = _OrderedDict(
          [('alpha',   [0.01, 0.5, 1.0]),
           ('l1_ratio', [.2, .4, .6, .8, 1.0]),
           ('normalize', [True, False])])
        _defaults[LinearRegression] = _OrderedDict(
          [('normalize', [True, False])])
        _defaults[GradientBoostingClassifier] = _OrderedDict(
          [('loss', ['deviance']),
           ('learning_rate', [0.01, .05, .1, .15, .2, .5]),
           ('n_estimators', [10, 25, 50, 100, 250]),
           ('max_depth', [3, 4, 5, 6, 8, 10, 12]),
           ('min_samples_split', [2, 3, 5]),
           # ('min_weight_fraction_leaf', [0.0]),
           ('subsample', [.8, .9, 1.0]),
           ('max_features', ['auto', 'sqrt', None])])
        _defaults[GradientBoostingRegressor] = _OrderedDict(
          [('loss', ['ls', 'lad', 'huber']),
           ('learning_rate', [0.01, .05, .1, .15, .2, .5]),
           ('n_estimators', [10, 25, 50, 100, 250]),
           ('max_depth', [3, 4, 5, 6, 8, 10, 12]),
           ('min_samples_split', [2, 3, 5]),
           ('min_samples_leaf', [2, 3, 4, 5]),
           ('subsample', [.8, .9, 1.0]),
           ('max_features', ['auto', 'sqrt', None])])
        _defaults[RandomForestRegressor] = _OrderedDict(
          [('n_estimators', [10, 25, 50, 100, 250]),
           # ('criterion', ['gini']),
           ('max_features', ['auto', 'sqrt', None]),
           ('max_depth', [3, 4, 5, 6, 8, 10, 12]),
           ('min_samples_split', [2, 3, 5]),
           ('min_samples_leaf', [2, 3, 4, 5]),
           ('bootstrap', [True, False])])
        _defaults[RandomForestClassifier] = _OrderedDict(
          [('n_estimators', [10, 25, 50, 100, 250]),
           ('criterion', ['gini']),
           ('max_features', ['auto', 'sqrt', None]),
           ('max_depth', [3, 4, 5, 6, 8, 10, 12]),
           ('min_samples_split', [2, 3, 5]),
           ('min_samples_leaf', [2, 3, 4, 5]),
           ('bootstrap', [True, False])])

    if model_factory not in _defaults:
        raise ValueError('Provided model_factory %s not currently supported for'
            ' automatic model parameter search. For a list of supported models '
            ' check graphlab.model_parameter_search.create documentation.'
            ' You may also create custom model factories for use with graphlab.random_search'
            ' or graphlab.grid_search.' % model_factory)

    return _defaults[model_factory]


