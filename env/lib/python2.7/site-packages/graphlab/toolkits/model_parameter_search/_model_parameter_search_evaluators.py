from graphlab.toolkits.recommender.util import _Recommender as _RecommenderModel
from graphlab.toolkits._supervised_learning import Classifier as _ClassifierModel
from graphlab.toolkits._supervised_learning import SupervisedLearningModel as _SupervisedLearningModel
from graphlab.toolkits.topic_model.topic_model import TopicModel as _TopicModel


def default_recommender_evaluator(model, training_set, validation_set):
    """
    Default evaluator for recommender models.

    Return a dictionary containing the following fields for training data:
    - training_precision @ 5
    - training_recall @ 5
    - training_rmse (if available)

    If `validation_set` is provided, the following fields will be included
    in the returned dictionary
    - validation_precision @ 5
    - validation_recall @ 5
    - validation_rmse (if avaiable)

    Returns
    -------
    out: dict
    """
    assert isinstance(model, _RecommenderModel), 'Evaluator expects a recommender model, actual model type is %s' % type(model)
    ret = {}
    evaluate_rmse = model['target'] is not None

    if 'training_rmse' in model.list_fields():
        ret['training_rmse'] = model['training_rmse']

    training_precision_recall = model.evaluate_precision_recall(training_set, exclude_known=False)
    for row in training_precision_recall['precision_recall_overall'][4:5]:
        ret['training_precision@' + str(row['cutoff'])] = row['precision']
        ret['training_recall@' + str(row['cutoff'])] = row['recall']

    if validation_set is not None:
        if evaluate_rmse:
            ret['validation_rmse'] = model.evaluate_rmse(validation_set, target=model['target'])['rmse_overall']
        validation_precision_recall = model.evaluate_precision_recall(validation_set, exclude_known=True)
        for row in validation_precision_recall['precision_recall_overall'][4:5]:
            ret['validation_precision@' + str(row['cutoff'])] = row['precision']
            ret['validation_recall@' + str(row['cutoff'])] = row['recall']
    return ret


def default_classifier_evaluator(model, training_set, validation_set):
    """
    Default evaluator for classifier models.

    Return a dictionary containing the following fields:
    - training_accuracy
    - validation_accuracy (if validation_set is avaiable)

    Returns
    -------
    out: dict
    """
    assert isinstance(model, _ClassifierModel), 'Evaluator expects a classifier model, actual model type is %s' % type(model)
    ret = {}

    # exception for svm which does not have 'training_accruacy'
    if 'training_accuracy' in model.list_fields():
        ret['training_accuracy'] = model['training_accuracy']
    else:
        ret['training_accuracy'] = model.evaluate(training_set, 'accuracy')['accuracy']

    if (validation_set):
        ret['validation_accuracy'] = model.evaluate(validation_set, 'accuracy')['accuracy']
    return ret


def default_regression_evaluator(model, training_set, validation_set):
    """
    Default evaluator for regression models.

    Return a dictionary containing the following fields:
    - training_rmse
    - validation_rmse (if validation_set is avaiable)

    Returns
    -------
    out: dict
    """
    # we don't have a base class for _RegressionModel....
    is_regression_model = isinstance(model, _SupervisedLearningModel) and not isinstance(model, _ClassifierModel)
    assert is_regression_model, 'Evaluator expect a regression model, actual model type is %s' % type(model)
    ret = {}
    ret['training_rmse'] = model['training_rmse']
    if (validation_set):
        ret['validation_rmse'] = model.evaluate(validation_set)['rmse']
    return ret


def default_topic_model_evaluator(model, training_set, validation_set):
    """
    Default evaluator for topic models.

    Return a dictionary containing the following fields:
    - perplexity

    If `validation_set` is not set, perplexity is set to None.

    Returns
    -------
    out: dict
    """
    assert isinstance(model, _TopicModel), 'Evaluator expect a topic model, actual model type is %s' % type(model)
    ret = {'perplexity': None}
    if validation_set is not None:
        ret['perplexity'] = model.evaluate(training_set, validation_set)['perplexity']
    return ret

def default_sklearn_evaluator(model, training_set, validation_set):
    ret = {}

    (X, y) = training_set
    ret['training_score'] = model.score(X, y)

    if validation_set is not None:
        (X, y) = validation_set
        ret['validation_score'] = model.score(X, y)

    return ret

def default_evaluator(model, training_set, validation_set):
    """
    Default evaluator for model parameter search.

    Select the appropriate default evaluator based on
    the model type. Supported model types include:
    recommender, classifier, regression, and topic model.

    For other model type, use a dummy evaluator that always
    returns None.
    """
    if (isinstance(model, _RecommenderModel)):
        return default_recommender_evaluator(model, training_set, validation_set)
    elif (isinstance(model, _ClassifierModel)):
        return default_classifier_evaluator(model, training_set, validation_set)
    elif (isinstance(model, _SupervisedLearningModel)):
        return default_regression_evaluator(model, training_set, validation_set)
    elif (isinstance(model, _TopicModel)):
        return default_topic_model_evaluator(model, training_set, validation_set)
    elif 'sklearn' in str(type(model)):
        return default_sklearn_evaluator(model, training_set, validation_set)
    else:
        return None
