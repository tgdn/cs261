"""
This package contains methods for comparing predictive machine
learning models.  The comparison result SFrame can be visualized in Canvas.
"""

from collections import OrderedDict as _OrderedDict

import graphlab.connect as _mt
import graphlab as _graphlab
import graphlab.toolkits.recommender.util as _recommender_util
from graphlab.canvas.inspect import _find_variable_name
from graphlab.toolkits.recommender.util import _Recommender
from graphlab.toolkits._supervised_learning import Classifier as _Classifier
from graphlab.toolkits._model import CustomModel as _CustomModel

def show_comparison(model_comp, models):
    r"""
    Visualizes the model comparison result (see compare) in Canvas.

    .. warning::

        This currently only works on Recommender type models.


    Parameters
    ----------
    model_comp : SFrame
        The dataset to use for model evaluation.

    models : list[models]
        List of trained models of single type.
        Currently, only Recommender is supported.


    Examples
    --------
    Generate a model comparison on a Recommender model type:

    >>> import graphlab
    >>> train_data = graphlab.SFrame({'user_id': ["0", "0", "0", "1", "1", "2", "2", "2"],
    ...                               'item_id': ["a", "c", "e", "b", "f", "b", "c", "d"]})
    >>> test_data = graphlab.SFrame({'user_id': ["0", "0", "1", "1", "1", "2", "2"],
    ...                              'item_id': ["b", "d", "a", "c", "e", "a", "e"]})
    >>> m1 = graphlab.item_similarity_recommender.create(train_data)
    >>> m2 = graphlab.item_similarity_recommender.create(train_data, only_top_k=1)
    >>> model_comp = graphlab.compare(test_data, [m1, m2])

    To view the model comparison do:

    >>> graphlab.show_comparison(model_comp, [m1, m2])
    """
    if type(model_comp) != _graphlab.SFrame:
        raise TypeError('"model_comp" must be a non empty SFrame')
    if type(models) != list or not all(map(lambda m: isinstance(m, _Recommender), models)):
        raise TypeError('"models" must be a list with Model elements')
    if len(models) == 0:
        raise ValueError('Length of "models" must have length greater than zero.')
    if len(models) != model_comp.num_rows():
        raise ValueError('Number of models must equal number of rows in "model_comp"')
    if not set(['metric', 'results','model']).issubset(set(model_comp.column_names())):
        raise ValueError('SFrame must be return value of graphlab.toolkits.compare_models()')

    model_dict = { row['model']: row['results'] for row in model_comp }

    ret = {}
    ret['dataset_label'] = model_comp['dataset'][0]
    ret['evaluations'] = model_dict

    for i in range(0, len(models)):
        models[i]._get_workflow().add_comparison(ret)

    return models[0].show(view='Comparison')

def compare(dataset, models, **kwargs):
    r"""
    Compare the prediction (or model-equivalent action) performance of models
    on a common test dataset.


    .. warning::

        This currently only works on Recommender type models.


    Parameters
    ----------
    dataset : SFrame
        The dataset to use for model evaluation.

    models : list[ models]
        List of trained models.


    Returns
    -------
    out : list[SFrame]
        A list of results where each one is an sframe of evaluation results of
        the respective model on the given dataset

    Examples
    --------
    If you have created two ItemSimilarityRecommenders ``m1`` and ``m2`` and have
    an :class:`~graphlab.SFrame` ``test_data``, then you may compare the
    performance of the two models on test data using:

    >>> import graphlab
    >>> train_data = graphlab.SFrame({'user_id': ["0", "0", "0", "1", "1", "2", "2", "2"],
    ...                               'item_id': ["a", "c", "e", "b", "f", "b", "c", "d"]})
    >>> test_data = graphlab.SFrame({'user_id': ["0", "0", "1", "1", "1", "2", "2"],
    ...                              'item_id': ["b", "d", "a", "c", "e", "a", "e"]})
    >>> m1 = graphlab.item_similarity_recommender.create(train_data)
    >>> m2 = graphlab.item_similarity_recommender.create(train_data, only_top_k=1)
    >>> model_comp = graphlab.compare(test_data, [m1, m2])

    The evaluation metric is automatically set to 'precision_recall', and the
    evaluation will be based on recommendations that exclude items seen in the
    training data.

    If you want to evaluate on the original training set:

    >>> model_comp = graphlab.compare(train_data, [m1, m2])

    Suppose you have four models, two trained with a target rating column, and
    the other two trained without a target. By default, the models are put into
    two different groups with "rmse", and "precision-recall" as the evaluation
    metric respectively.

    >>> train_data2 = graphlab.SFrame({'user_id': ["0", "0", "0", "1", "1", "2", "2", "2"],
    ...                                'item_id': ["a", "c", "e", "b", "f", "b", "c", "d"],
    ...                                'rating': [1, 3, 4, 5, 3, 4, 2, 5]})
    >>> test_data2 = graphlab.SFrame({'user_id': ["0", "0", "1", "1", "1", "2", "2"],
    ...                               'item_id': ["b", "d", "a", "c", "e", "a", "e"],
    ...                               'rating': [3, 5, 4, 4, 3, 5, 2]})
    >>> m3 = graphlab.factorization_recommender.create(train_data2, target='rating')
    >>> m4 = graphlab.factorization_recommender.create(train_data2, target='rating')
    >>> model_comp = graphlab.compare(test_data2, [m3, m4])

    To compare all four models, you can do:

    >>> model_comp = graphlab.compare(test_data2, [m1, m2, m3, m4])
    """

    _mt._get_metric_tracker().track('toolkit.compare_models')

    num_models = len(models)

    model_names = []
    results = []

    if num_models < 1:
        raise ValueError("Must pass in at least one model to \
                           evaluate")

    base_model_types = map(lambda m: m.__class__.__base__, models)
    unique_base_model_types = list(set(base_model_types))

    if len(unique_base_model_types) != 1:
        raise ValueError("Must pass in related model types")
    metric='precision_recall'
    if unique_base_model_types[0] == _Recommender:
        results = __evaluate_recomenders(dataset, models, metric=metric, **kwargs)
    else:
        import logging
        logging.warn('Model comparison currently only supports comparing models created by graphlab.toolkits.recommender or its subclasses such as item_similarity_recommender.')
        return _graphlab.SFrame()

    print("Model compare metric: " + metric)

    metric_label = metric
    if metric == 'precision_recall':
        metric_label = 'precision_recall_overall'

    #cast result SFrames to lists
    results = map( lambda x: {metric: list(x[metric_label]) }, results)

    for i in range(0,len(models)):
        name = _find_variable_name(models[i])[0]
        if name is None or not isinstance(name,str):
            name = 'Model_'+str(i)
        model_names.append(name)

    model_types = map(lambda m: m.__class__.__name__, models)
    dataset_label = _find_variable_name(dataset)[0]
    dataset_labels = [dataset_label] * num_models
    metrics = [metric] * num_models

    sframe_results = _graphlab.SFrame({'model': model_names,
                                        'dataset': dataset_labels,
                                        'model_type': list(model_types),
                                        'metric': metrics,
                                        'results': list(results)
                                        })

    return sframe_results

def __evaluate_recomenders(dataset, models, metric, **kwargs):
    return _recommender_util.compare_models(dataset, models,
                                                    metric=metric,
                                                    **kwargs)

def __evaluate_classifiers(dataset, models, metric, **kwargs):
    results = []

    for m in models:
        # TODO this attaches the evaluation to the model with the wrong dataset label.
        # How can this be attached with the correct label?
        results.append(m.evaluate(dataset,
                       metric=metric,
                       **kwargs) )
    return results
