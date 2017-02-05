import graphlab.connect as _mt
import graphlab as _graphlab
import random as _random
import array as _array
import logging as _logging
import time as _time
import six as _six
from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits._model import Model as _Model
from graphlab.toolkits._internal_utils import _toolkit_repr_print, \
                                        _precomputed_field
from graphlab.toolkits._model_workflow import _collect_model_workflow
import graphlab.aggregate as _Aggregate
from graphlab.data_structures.sarray import SArray as _SArray
from graphlab.data_structures.sframe import SFrame as _SFrame
from graphlab.deps import numpy as _numpy, HAS_NUMPY as _HAS_NUMPY

from graphlab._beta.views import RecommenderEvaluateView as _RecommenderEvaluateView
from graphlab._beta.views import RecommenderExploreView as _RecommenderExploreView
from graphlab._beta.views import RecommenderDescriptionView as _model_description
from graphlab._beta.views import OverviewApp as _OverviewApp

def _create(observation_data,
           user_id='user_id', item_id='item_id', target=None,
           user_data=None, item_data=None,
           ranking=True,
           verbose=True):
    r"""
    A unified interface for training recommender models. Based on simple
    characteristics of the data, a type of model is selected and trained. The
    trained model can be used to predict ratings and make recommendations.

    To use specific options of a desired model, use the ``create`` function
    of the corresponding model.

    Parameters
    ----------
    observation_data : SFrame
        The dataset to use for training the model. It must contain a column of
        user ids and a column of item ids. Each row represents an observed
        interaction between the user and the item.  The (user, item) pairs
        are stored with the model so that they can later be excluded from
        recommendations if desired. It can optionally contain a target ratings
        column. All other columns are interpreted by the underlying model as
        side features for the observations.

        The user id and item id columns must be of type 'int' or 'str'. The
        target column must be of type 'int' or 'float'.

    user_id : string, optional
        The name of the column in `observation_data` that corresponds to the
        user id.

    item_id : string, optional
        The name of the column in `observation_data` that corresponds to the
        item id.

    target : string, optional
        Name of the column in `observation_data` containing ratings given by
        users to items, if applicable.

    user_data : SFrame, optional
        Side information for the users. This SFrame must have a column with the
        same name as what is specified by the `user_id` input parameter.
        `user_data` can provide any amount of additional user-specific
        information.

    item_data : SFrame, optional
        Side information for the items. This SFrame must have a column with the
        same name as what is specified by the `item_id` input parameter.
        `item_data` can provide any amount of additional item-specific
        information.

    ranking : bool, optional
        Determine whether or not the goal is to rank items for each user.

    verbose : bool, optional
        Enables verbose output.

    Returns
    -------
    out :  A trained model.
        - If a target column is given, then
          :class:`graphlab.recommender.factorization_recommender.FactorizationRecommender`.
        - If no target column is given, then
          :class:`graphlab.recommender.item_similarity_recommender.ItemSimilarityRecommender`.

    Examples
    --------
    **Basic usage**

    Given basic user-item observation data, an
    :class:`~graphlab.recommender.item_similarity_recommender.ItemSimilarityRecommender` is created:

    >>> sf = graphlab.SFrame({'user_id': ['0', '0', '0', '1', '1', '2', '2', '2'],
    ...                       'item_id': ['a', 'b', 'c', 'a', 'b', 'b', 'c', 'd']})
    >>> m = graphlab.recommender.create(sf)
    >>> recs = m.recommend()

    **Creating a model for ratings data**

    This trains a :class:`~graphlab.recommender.factorization_recommender.FactorizationRecommender` that
    can predict target ratings:

    >>> sf2 = graphlab.SFrame({'user_id': ['0', '0', '0', '1', '1', '2', '2', '2'],
    ...                        'item_id': ['a', 'b', 'c', 'a', 'b', 'b', 'c', 'd'],
    ...                        'rating': [1, 3, 2, 5, 4, 1, 4, 3]})
    >>> m2 = graphlab.recommender.create(sf2, target="rating", ranking = False)

    **Creating specific models**

    Specific models allow for a number of additional options during create.
    The available recommenders are all in the graphlab.recommender namespace.
    For the complete list of acceptable options, please refer to the documentation
    for individual models. Such options can be passed to the underlying model
    just like any other parameter. For example, the following code creates
    an :class:`~graphlab.recommender.ItemSimilarityRecommender` with a space-saving
    option called `only_top_k`. The returned model stores only the 2 most
    similar items for item:

    >>> from graphlab.recommender import item_similarity_recommender
    >>> item_similarity_recommender.create(sf, only_top_k=2)
    """

    if not ranking:
        _mt._get_metric_tracker().track('toolkit.recsys.create.rating')
    else:
        _mt._get_metric_tracker().track('toolkit.recsys.create.ranking')

    if not (isinstance(observation_data, _SFrame)):
        raise TypeError('observation_data input must be a SFrame')

    side_data = (user_data is not None) or (item_data is not None)
    if user_data is not None:
        if not isinstance(user_data, _SFrame):
            raise TypeError('Provided user_data must be an SFrame.')
    if item_data is not None:
        if not isinstance(item_data, _SFrame):
            raise TypeError('Provided item_data must be an SFrame.')

    if target is None:
        if ranking:
            if side_data:
                method = 'ranking_factorization_recommender'
            else:
                method = 'item_similarity'
        else:
            if side_data:
                method = 'ranking_factorization_recommender'
            else:
                method = 'item_similarity'
    else:
        if ranking:
            if side_data:
                method = 'ranking_factorization_recommender'
            else:
                method = 'ranking_factorization_recommender'
        else:
            if side_data:
                method = 'factorization_recommender'
            else:
                method = 'factorization_recommender'

    opts = {'observation_data': observation_data,
            'user_id': user_id,
            'item_id': item_id,
            'target': target,
            'user_data': user_data,
            'item_data': item_data}


    if method == "item_similarity":
        return _graphlab.recommender.item_similarity_recommender.create(**opts)
    elif method == "factorization_recommender":
        return _graphlab.recommender.factorization_recommender.create(**opts)
    elif method == "ranking_factorization_recommender":
        return _graphlab.recommender.ranking_factorization_recommender.create(**opts)
    else:
        raise RuntimeError("Provided method not recognized.")


def __plot_histogram(measurements, means, names=None, metric_name=None):
    """
    Plot histograms of the measurements, overlaid with vertical lines
    representing the means of the measurements.

    Parameters
    ----------
    measurements : list
        List of measurements (recall, precision or RMSE).

    means : list
        List of doubles, intended to be the mean of each list in
        'measurements'.

    names : list
        List of model name strings.

    metric_name : string
        Name of the metric.
    """
    num_measurements = len(measurements)

    # A list of colors for plotting
    COLORS_LIST = ['b', 'g', 'r', 'k', 'm', 'c']
    colors = [COLORS_LIST[i % len(COLORS_LIST)] for i in range(num_measurements)]

    hist_handle = _pp.hist(measurements, bins=20,
                           color=colors,
                           label=names, hold=True)
    _pp.legend()
    # hist() returns a list of two lists, the first is a list of all the counts,
    # the second is a list of the bin centers.  We need the maximum count so we know
    # how tall the vertical line should be.  So we take the max of the max of the
    # first list of lists
    max_count = max([max(hcount) for hcount in hist_handle[0]])
    _pp.vlines(means, 0, max_count, colors=colors)
    _pp.xlabel(metric_name)
    _pp.ylabel('Counts')


def __plot_overlap_hists(results, label, names, bins=20, alpha=0.3):
    """
    Plot overlapping (un-normalized) histograms for a list of one-dimensional
    series.

    Parameters
    ----------
    results : list
        List of list-like objects. Each element is plotted as a separate histogram.

    label : string
        Label for the x-axis of the histogram.

    names : list
        Names for each series in `results'.

    bins : int
        Number of bins. Default is 20.

    alpha : float
        Opacity of the histogram patches. Default is 0.4.
    """
    assert _HAS_NUMPY, 'Cannot find numpy'
    fig, ax = _pp.subplots()

    # plot first series to fix the bins
    counts, bins, patches = ax.hist(results[0], bins=bins, alpha=alpha, lw=0.1,
                                    label=names[0])
    clr = patches[0].get_facecolor()
    counts = _numpy.insert(counts, 0, 0)
    ax.step(bins, counts, color=clr, lw=5)

    # plot the rest of the series
    if len(results) > 1:
        for series, name in zip(results[1:], names[1:]):
            counts, bins, patches = ax.hist(series, bins=bins, alpha=alpha, lw=0.03,
                                            label=name, fill=True)
            clr = patches[0].get_facecolor()
            counts = _numpy.insert(counts, 0, 0)
            ax.step(bins, counts, color=clr, lw=4)

    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax.set_xlabel(label)
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.show()


def _compare_results_precision_recall(results, model_names=None):
    """
    Compare models that output precision/recall. Display the per-user
    precision/recall scatter plot, the histograms of precision, and the
    histograms of recall.

    Parameters
    ----------
    results : list
        List of SFrames. Each SFrame describes the evaluation results for a
        separate model.

    model_names : list
        List of model name strings.
    """

    num_models = len(results)
    COLORS_LIST = ['b', 'g', 'r', 'k', 'm', 'c']

    if num_models < 1:
        return

    if model_names is None:
        model_names = ["model {}".format(i) for i in range(num_models)]

    pr_curves_by_model = [res['precision_recall_overall'] for res in results]
    fig, ax = _pp.subplots()

    for i in range(num_models):
        pr_curve = pr_curves_by_model[i]
        pr_curve = pr_curve.sort('recall')
        name = 'Model ' + str(i + 1)
        if model_names is not None:
            name = model_names[i]

        ax.plot(list(pr_curve['recall']), list(pr_curve['precision']),
                COLORS_LIST[i % len(COLORS_LIST)], label=name)

    ax.set_title('Precision-Recall Averaged Over Users')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    fig.show()


def _compare_results_rmse2(results, model_names=None):
    """
    Compare models that output root-mean-squared error (RMSE).

    Parameters
    ----------
    results : list
        List of dataframes describing evaluation results for each model.

    model_names : list
        List of model name strings.
    """
    assert _HAS_NUMPY, 'Cannot find numpy'
    ## Validate the results
    num_models = len(results)

    if num_models < 1 or num_models > len(_rcParams['axes.color_cycle']):
        return

    if model_names is None:
        model_names = ["model {}".format(i) for i in range(num_models)]

    ## Histograms of per-user and per-item RMSE
    user_rmse_by_model = [list(elem['rmse_by_user']['rmse']) for elem in results]
    __plot_overlap_hists(user_rmse_by_model, 'Per-User RMSE', model_names, bins=100)

    item_rmse_by_model = [list(elem['rmse_by_item']['rmse']) for elem in results]
    __plot_overlap_hists(item_rmse_by_model, 'Per-Item RMSE', model_names, bins=100)

    ## Bar chart of Overall RMSE by model
    overall_rmse_by_model = [elem['rmse_overall'] for elem in results]

    fig, ax = _pp.subplots()
    BAR_WIDTH = 0.3
    centers = _numpy.arange(num_models) + BAR_WIDTH
    ax.bar(centers, overall_rmse_by_model, BAR_WIDTH)
    ax.set_xticks(centers + BAR_WIDTH / 2)
    ax.set_xticklabels(model_names)
    ax.set_title('Overall RMSE')
    fig.show()


def compare_models(dataset, models, model_names=None, user_sample=1.0,
                   metric='auto',
                   target=None,
                   exclude_known_for_precision_recall=True,
                   make_plot=False,
                   verbose=True,
                   **kwargs):
    """
    Compare the prediction or recommendation performance of recommender models
    on a common test dataset.

    Models that are trained to predict ratings are compared separately from
    models that are trained without target ratings.  The ratings prediction
    models are compared on root-mean-squared error, and the rest are compared on
    precision-recall.

    Parameters
    ----------
    dataset : SFrame
        The dataset to use for model evaluation.

    models : list[recommender models]
        List of trained recommender models.

    model_names : list[str], optional
        List of model name strings for display.

    user_sample : float, optional
        Sampling proportion of unique users to use in estimating model
        performance. Defaults to 1.0, i.e. use all users in the dataset.

    metric : str, {'auto', 'rmse', 'precision_recall'}, optional
        Metric for the evaluation. The default automatically splits
        models into two groups with their default evaluation metric respectively:
        'rmse' for models trained with a target, and 'precision_recall'
        otherwise.

    target : str, optional
        The name of the target column for evaluating rmse. If the model is
        trained with a target column, the default is to using the same column.
        If the model is trained without a target column and `metric='rmse'`,
        then this option must be provided by user.

    exclude_known_for_precision_recall : bool, optional
        A useful option when `metric='precision_recall'`. Recommender models
        automatically exclude items seen in the training data from the
        final recommendation list. If the input evaluation `dataset` is the
        same as the data used for training the models, set this option to False.


    verbose : bool, optional
        If true, print the progress.

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
    >>> graphlab.recommender.util.compare_models(test_data, [m1, m2], model_names=["m1", "m2"])

    The evaluation metric is automatically set to 'precision_recall', and the
    evaluation will be based on recommendations that exclude items seen in the
    training data.

    If you want to evaluate on the original training set:

    >>> graphlab.recommender.util.compare_models(train_data, [m1, m2],
    ...                                     exclude_known_for_precision_recall=False)

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
    >>> graphlab.recommender.util.compare_models(test_data2, [m3, m4])

    To compare all four models using the same 'precision_recall' metric, you can
    do:

    >>> graphlab.recommender.util.compare_models(test_data2, [m1, m2, m3, m4],
    ...                                          metric='precision_recall')
    """
    _mt._get_metric_tracker().track('toolkit.recsys.compare_models')

    num_models = len(models)

    if model_names is None:
        model_names = ['M' + str(i) for i in range(len(models))]

    if num_models < 1:
        raise ValueError("Must pass in at least one recommender model to \
                           evaluate")

    if model_names is not None and len(model_names) != num_models:
        raise ValueError("Must pass in the same number of model names as \
                          models")

    # if we are asked to sample the users, come up with a list of unique users
    if user_sample < 1.0:
        user_id_name = models[0].get('user_id')
        if user_id_name is None:
            raise ValueError("user_id not set in model(s)")
        user_sa = dataset[user_id_name]
        unique_users = list(user_sa.unique())
        nusers = len(unique_users)
        ntake = int(round(user_sample * nusers))

        _random.shuffle(unique_users)

        users = unique_users[:ntake]
        print("compare_models: using " + str(ntake) + \
              " users to estimate model performance")
        users = frozenset(users)
        ix = [u in users for u in dataset[user_id_name]]
        dataset_subset = dataset[_SArray(ix) == True]
    else:
        dataset_subset = dataset

    results = []
    for (m, mname) in zip(models, model_names):
        if verbose:
            print('PROGRESS: Evaluate model %s' % mname)
        r = m.evaluate(dataset_subset,
                       metric,
                       exclude_known_for_precision_recall,
                       target,
                       verbose=verbose,
                       cutoffs=list(range(1,11,1))+list(range(11,50,5)),
                       **kwargs)
        results.append(r)

    return results


def precision_recall_by_user(observed_user_items,
                             recommendations,
                             cutoffs=[10]):
    """
    Compute precision and recall at a given cutoff for each user. In information
    retrieval terms, precision represents the ratio of relevant, retrieved items
    to the number of relevant items. Recall represents the ratio of relevant,
    retrieved items to the number of relevant items.

    Let :math:`p_k` be a vector of the first :math:`k` elements in the
    recommendations for a particular user, and let :math:`a` be the set of items
    in ``observed_user_items`` for that user. The "precision at cutoff k" for
    this user is defined as

    .. math::
        P(k) = \\frac{ | a \cap p_k | }{k},

    while "recall at cutoff k" is defined as

    .. math::
        R(k) = \\frac{ | a \cap p_k | }{|a|}

    The order of the elements in the recommendations affects the returned
    precision and recall scores.

    Parameters
    ----------
    observed_user_items : SFrame
        An SFrame containing observed user item pairs, where the first
        column contains user ids and the second column contains item ids.

    recommendations : SFrame
        An SFrame containing columns pertaining to the user id, the item id,
        the score given to that pair, and the rank of that item among the
        recommendations made for user id. For example, see the output of
        recommend() produced by any graphlab.recommender model.

    cutoffs : list[int], optional
        The cutoffs to use when computing precision and recall.

    Returns
    -------
    out : SFrame
        An SFrame containing columns user id, cutoff, precision, recall, and
        count where  the precision and recall are reported for each user at
        each requested cutoff, and count is the number of observations for
        that user id.

    Notes
    -----
    The corner cases that involve empty lists were chosen to be consistent
    with the feasible set of precision-recall curves, which start at
    (precision, recall) = (1,0) and end at (0,1). However, we do not believe
    there is a well-known concensus on this choice.

    Examples
    --------
    Given SFrames ``train_data`` and ``test_data`` with columns user_id
    and item_id:

    >>> from graphlab.toolkits.recommender.util import precision_recall_by_user
    >>> m = graphlab.recommender.create(train_data)
    >>> recs = m.recommend()
    >>> precision_recall_by_user(test_data, recs, cutoffs=[5, 10])
    """
    _mt._get_metric_tracker().track('toolkit.recsys.precision_recall_by_user')

    assert type(observed_user_items) == _SFrame
    assert type(recommendations) == _SFrame
    assert type(cutoffs) == list
    assert min(cutoffs) > 0, "All cutoffs must be positive integers."
    assert recommendations.num_columns() >= 2
    user_id = recommendations.column_names()[0]
    item_id = recommendations.column_names()[1]

    assert observed_user_items.num_rows() > 0, \
           "Evaluating precision and recall requires a non-empty " + \
           "observed_user_items."
    assert user_id in observed_user_items.column_names(), \
            "User column required in observed_user_items."
    assert item_id in observed_user_items.column_names(), \
            "Item column required in observed_user_items."
    assert observed_user_items[user_id].dtype() == \
           recommendations[user_id].dtype(), \
           "The user column in the two provided SFrames must have the same type."
    assert observed_user_items[item_id].dtype() == \
           recommendations[item_id].dtype(), \
           "The user column in the two provided SFrames must have the same type."

    cutoffs = _array.array('f', cutoffs)

    opts = {'data': observed_user_items,
            'recommendations': recommendations,
            'cutoffs': cutoffs}

    response = _graphlab.toolkits._main.run('evaluation_precision_recall_by_user', opts)
    sf = _SFrame(None, _proxy=response['pr'])
    return sf.sort([user_id, 'cutoff'])


def random_split_by_user(dataset,
                         user_id='user_id',
                         item_id='item_id',
                         max_num_users=1000,
                         item_test_proportion=.2,
                         random_seed=0):
    """Create a recommender-friendly train-test split of the provided data set.

    The test dataset is generated by first choosing `max_num_users` out of the
    total number of users in `dataset`. Then, for each of the chosen test users,
    a portion of the user's items (determined by `item_test_proportion`) is
    randomly chosen to be included in the test set. This split allows the
    training data to retain enough information about the users in the testset,
    so that adequate recommendations can be made. The total number of users
    in the test set may be fewer than `max_num_users` if a user was chosen for
    the test set but none of their items are selected.

    Parameters
    ----------
    dataset : SFrame
        An SFrame containing (user, item) pairs.

    user_id : str, optional
        The name of the column in ``dataset`` that contains user ids.

    item_id : str, optional
        The name of the column in ``dataset`` that contains item ids.

    max_num_users : int, optional
        The maximum number of users to use to construct the test set. If
        set to 'None', then use all available users.

    item_test_proportion : float, optional
        The desired probability that a test user's item will be chosen
        for the test set.

    random_seed : int, optional The random seed to use for
        randomization.  If None, then the random seed is different
        every time; if numeric, then subsequent calls with the same
        dataset and random seed with have the same split.

    Returns
    -------
    train, test : SFrame
        A tuple with two datasets to be used for training and testing.

    Examples
    --------
    >>> import graphlab as gl
    >>> sf = gl.SFrame('https://static.turi.com/datasets/audioscrobbler')
    >>> train, test = gl.recommender.util.random_split_by_user(sf, max_num_users=100)

    """

    _mt._get_metric_tracker().track('toolkit.recsys.random_split_by_user')

    assert user_id in dataset.column_names(), \
        'Provided user column "{0}" not found in data set.'.format(user_id)
    assert item_id in dataset.column_names(), \
        'Provided item column "{0}" not found in data set.'.format(item_id)

    if max_num_users == 'all':
        max_num_users = None

    if random_seed is None:
        import time
        random_seed = int(hash("%20f" % time.time()) % 2**63)

    opts = {'dataset': dataset,
            'user_id': user_id,
            'item_id': item_id,
            'max_num_users': max_num_users,
            'item_test_proportion': item_test_proportion,
            'random_seed': random_seed}

    response = _graphlab.toolkits._main.run('recsys_train_test_split', opts)
    train = _SFrame(None, _proxy=response['train'])
    test = _SFrame(None, _proxy=response['test'])
    return train, test

class RecommenderViews(object):

    def __init__(self, model):
        self._model = model

    def __repr__(self):
        title = ['Available views for this RecommenderModel']
        title += ["--------------------------------------------------"]
        title += ["explore:  Explore the model qualitatively."]
        title += ["evaluate: Understand model performance quantitatively."]
        return '\n'.join(title)

    def overview(self,
                validation_set,
                baseline=None,
                observation_data=None,
                user_data=None,
                item_data=None,
                user_name_column=None,
                user_description_column=None,
                user_url_column=None,
                item_name_column=None,
                item_description_column=None,
                item_url_column=None):
        """
        Creates a visualization of the performance of this model relative to
        a baseline model, and a visualization of the model that helps explore
        and qualitatively evaluate the recommendations made by the model,
        deploys both to a local ViewServer, and returns the resulting view.

        Parameters
        ----------

        validation_set : SFrame
            The dataset to use for evaluating the model. It must contain a column of
            user ids and a column of item ids having the same column names that
            the model used during training.

        baseline : RecommenderModel, optional
            A second model to use as a comparison. If not provided, then a
            PopularityRecommender is created.

        observation_data : SFrame
            The dataset to use for training the model. It must contain a column of
            user ids and a column of item ids. Each row represents an observed
            interaction between the user and the item.

        user_data : SFrame
            A dataset where each row contains data about a single user.
            This SFrame must have a column with the same name as what is
            specified by the `user_id` input parameter when creating the model.

        user_name_column : str, optional
            The name of the column in the `user_data` dataset that contains
            the name of the item. These names are used in the interactive
            visualization.

        user_description_column : str, optional
            The name of the column in the `user_data` dataset that contains
            a description of the item. These names are used in the interactive
            visualization.

        user_path_column : str, optional
            The name of the column in the `user_data` dataset that contains
            a path to an image of the item. These images are used in the
            interactive visualization.


        item_data : SFrame
            A dataset where each row contains data about a single item.
            This SFrame must have a column with the same name as what is
            specified by the `item_id` input parameter when creating the model.

        item_name_column : str, optional
            The name of the column in the `item_data` dataset that contains
            the name of the item. These names are used in the interactive
            visualization.

        item_description_column : str, optional
            The name of the column in the `item_data` dataset that contains
            a description of the item. These names are used in the interactive
            visualization.

        item_path_column : str, optional
            The name of the column in the `item_data` dataset that contains
            a path to an image of the item. These images are used in the
            interactive visualization.

        Returns
        -------
        out : View
            This object can be opened in a web browser with .show().

        Examples
        --------

        .. sourcecode:: python

            >>> import graphlab
            >>> filename = "https://static.turi.com/datasets/movie_ratings/training_data.csv"
            >>> data = graphlab.SFrame.read_csv(filename)

            # Make a separate data set containing item titles
            >>> items = data.groupby('movie', graphlab.aggregate.MEAN('rating'))
            >>> items = items.add_row_number('item_id')
            >>> data = data.join(items, on='movie')

            # Create a holdout set for better evaluation
            >>> train, valid = graphlab.recommender.util.random_split_by_user(data, 'user', 'item_id')

            # Train a model
            >>> model = graphlab.recommender.create(train, 'user', 'item_id')

            # Create an interactive view
            >>> view = model.views.overview(
                    validation_set=valid,
                    item_data=items,
                    item_name_column='movie')
            >>> view.show()
        """
        _mt._get_metric_tracker().track('toolkits.recommender.overview')
        return _OverviewApp(
            self.description(),
            self.explore(
                observation_data=observation_data,
                user_data=user_data,
                item_data=item_data,
                user_name_column=user_name_column,
                user_description_column=user_description_column,
                user_url_column=user_url_column,
                item_name_column=item_name_column,
                item_description_column=item_description_column,
                item_url_column=item_url_column),
            self.evaluate(
                validation_set=validation_set,
                baseline=baseline),
            title="Recommender View")

    def description(self):
        """

        Create a visualization of a description of the model: model type,
        and information about the training data.

        Returns
        -------
        out : View

        .. sourcecode:: python

            >>> view = model.views.description()
            >>> view.show()
        """
        return _model_description(self._model)


    def evaluate(self, validation_set, baseline=None):
        """
        Create a visualization of the performance of this model relative to
        a baseline model.

        Parameters
        ----------

        validation_set : SFrame
            The dataset to use for evaluating the model. It must contain a column of
            user ids and a column of item ids having the same column names that
            the model used during training.

        baseline : RecommenderModel, optional
            A second model to use as a comparison. If not provided, then a
            PopularityRecommender is created.

        Returns
        -------
        out : View

        Examples
        --------

        .. sourcecode:: python

            >>> filename = "https://static.turi.com/datasets/movie_ratings/training_data.csv"
            >>> data = graphlab.SFrame.read_csv(filename)

            # Make a separate data set containing item titles
            >>> items = data.groupby('movie', gl.aggregate.MEAN('rating'))
            >>> items = items.add_row_number('item_id')
            >>> data = data.join(items, on='movie')

            # Create a holdout set for better evaluation
            >>> train, valid = graphlab.recommender.util.random_split_by_user(data, 'user', 'item_id')

            # Train a model
            >>> model = graphlab.recommender.create(train, 'user', 'item_id')

            # Show an interactive view
            >>> view = model.views.evaluate(valid)
            >>> view.show()

        """
        _mt._get_metric_tracker().track('toolkits.recommender.evaluate')
        if baseline is None:
            baseline = self._model._get_popularity_baseline()

        models = {'recommender': self._model, 'baseline': baseline}
        return _RecommenderEvaluateView(validation_set, models)

    def explore(self,
                observation_data=None,
                user_data=None,
                item_data=None,
                user_name_column=None,
                user_description_column=None,
                user_url_column=None,
                item_name_column=None,
                item_description_column=None,
                item_url_column=None):
        """
        Create a visualization of a RecommenderModel that helps explore and
        qualitatively evaluate the recommendations made by the model.

        Parameters
        ----------

        observation_data : SFrame
            The dataset to use for training the model. It must contain a column of
            user ids and a column of item ids. Each row represents an observed
            interaction between the user and the item.

        user_data : SFrame
            A dataset where each row contains data about a single user.
            This SFrame must have a column with the same name as what is
            specified by the `user_id` input parameter when creating the model.

        user_name_column : str, optional
            The name of the column in the `user_data` dataset that contains
            the name of the item. These names are used in the interactive
            visualization.

        user_description_column : str, optional
            The name of the column in the `user_data` dataset that contains
            a description of the item. These names are used in the interactive
            visualization.

        user_path_column : str, optional
            The name of the column in the `user_data` dataset that contains
            a path to an image of the item. These images are used in the
            interactive visualization.


        item_data : SFrame
            A dataset where each row contains data about a single item.
            This SFrame must have a column with the same name as what is
            specified by the `item_id` input parameter when creating the model.

        item_name_column : str, optional
            The name of the column in the `item_data` dataset that contains
            the name of the item. These names are used in the interactive
            visualization.

        item_description_column : str, optional
            The name of the column in the `item_data` dataset that contains
            a description of the item. These names are used in the interactive
            visualization.

        item_path_column : str, optional
            The name of the column in the `item_data` dataset that contains
            a path to an image of the item. These images are used in the
            interactive visualization.


        Returns
        -------
        out : View
            This object can be visualized with .show().

        Examples
        --------

        .. sourcecode:: python

            >>> import graphlab
            >>> filename = "https://static.turi.com/datasets/movie_ratings/training_data.csv"
            >>> data = graphlab.SFrame.read_csv(filename)

            # Make a separate data set containing item titles
            >>> items = data.groupby('movie', graphlab.aggregate.MEAN('rating'))
            >>> items = items.add_row_number('item_id')
            >>> data = data.join(items, on='movie')

            # Train a model
            >>> model = graphlab.recommender.create(data, 'user', 'item_id')

            # Create an interactive view
            >>> view = model.views.explore(item_data=items, item_name_column='movie')
            >>> view.show()
        """
        _mt._get_metric_tracker().track('toolkits.recommender.explore')
        return _RecommenderExploreView(self._model,
                observation_data=observation_data,
                user_data=user_data,
                item_data=item_data,
                user_name_column=user_name_column,
                user_description_column=user_description_column,
                user_url_column=user_url_column,
                item_name_column=item_name_column,
                item_description_column=item_description_column,
                item_url_column=item_url_column)

class _Recommender(_Model):
    """
    The abstract class for GraphLab Create recommender system models. This class
    defines methods common to all recommender system models, but leaves unique
    model details to separate model classes.
    """
    @property
    def views(self):
        """
        Interactively visualize a recommender model.

        Once a model has been trained, you can easily visualize the model. There
        are three built-in visualizations to help explore, explain, and evaluate
        the model.

        Examples
        --------
        .. sourcecode:: python

            # Show an interactive view
            >>> view = model.views.evaluate(valid)
            >>> view.show()

            # Explore predictions
            >>> view = model.views.explore(item_data=items,
            ...              item_name_column='movie')

            # Explore evals
            >>> view = model.views.overview(
            ...     validation_set=valid,
            ...     item_data=items,
            ...     item_name_column='movie')
            >>> view.show()

        See Also
        --------
        graphlab.recommender.util.RecommenderViews
        """
        return RecommenderViews(self)

    @classmethod
    def _get_queryable_methods(cls):
        '''Returns a list of method names that are queryable through Predictive
        Service'''
        return {'predict': {
                    'dataset': 'sframe',
                    'new_observation_data': 'sframe',
                    'new_user_data': 'sframe',
                    'new_item_data': 'sframe'
                },
                'recommend': {
                    'users': ['sframe', 'sarray'],
                    'items': ['sframe', 'sarray'],
                    'new_observation_data': 'sframe',
                    'new_user_data': 'sframe',
                    'new_item_data': 'sframe',
                    'exclude': 'sframe'}
                }

    def _get_ml_metric_config(self):
        '''Returns a list of metric names that are computed through Predictive
        Service'''
        return {'endpoint_type': 'recommender', 'item_id': self.item_id}

    def list_fields(self):
        """
        Get the current settings of the model. The keys depend on the type of
        model.

        Returns
        -------
        out : list
            A list of fields that can be queried using the ``get`` method.
        """

        opts = {'model': self.__proxy__}
        response = _graphlab.toolkits._main.run('recsys_list_fields', opts)
        return [s for s in response['value'] if not s.startswith("_")]

    def get(self, field):
        """
        Get the value of a particular field.

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out
            The current value of the requested field.

        Examples
        --------
        >>> data = graphlab.SFrame({'user_id': ["0", "0", "0", "1", "1", "2", "2", "2"],
        ...                         'item_id': ["a", "b", "c", "a", "b", "b", "c", "d"],
        ...                         'rating': [1, 3, 2, 5, 4, 1, 4, 3]})
        >>> from graphlab.recommender
        >>> m = factorization_recommender.create(data, "user_id", "item_id", "rating")
        >>> d = m.get("coefficients")
        >>> U1 = d['user_id']
        >>> U2 = d['movie_id']
        """

        opts = {'model': self.__proxy__, 'field': field}
        response = _graphlab.toolkits._main.run('recsys_get_value', opts)

        def type_tr(v):
            if type(v) is dict:
                return dict( (k, type_tr(v)) for k, v in _six.iteritems(v))

            elif isinstance(v, _graphlab.cython.cy_sframe.UnitySFrameProxy):
                return _SFrame(None, _proxy=v)

            else:
                return v

        return type_tr(response["value"])

    def get_num_items_per_user(self):
        """
        Get the number of items observed for each user.

        Returns
        -------
        out : SFrame
            An SFrame with a column containing each observed user and another column
            containing the corresponding number of items observed during training.

        Examples
        --------

        """
        opts = {'model': self.__proxy__}
        response = _graphlab.toolkits._main.run('recsys_get_num_items_per_user', opts)
        return _SFrame(None, _proxy=response['data'])

    def get_num_users_per_item(self):
        """
        Get the number of users observed for each item.

        Returns
        -------
        out : SFrame
            An SFrame with a column containing each observed item and another column
            containing the corresponding number of items observed during training.

        Examples
        --------

        """
        opts = {'model': self.__proxy__}
        response = _graphlab.toolkits._main.run('recsys_get_num_users_per_item', opts)
        return _SFrame(None, _proxy=response['data'])


    def __str__(self):
        """
        Returns the type of model.

        Returns
        -------
        out : string
            The type of model.
        """
        return self.__class__.__name__

    def _get_summary_struct(self):
        """
        Returns a structured description of the model, including (where relevant)
        the schema of the training data, description of the training data,
        training statistics, and model hyperparameters.

        Returns
        -------
        sections : list (of list of tuples)
            A list of summary sections.
              Each section is a list.
                Each item in a section list is a tuple of the form:
                  ('<label>','<field>')
        section_titles: list
            A list of section names.
              The order matches that of the 'sections' object.
        """

        stats = self.list_fields()
        options = self.get_current_options()

        section_titles = []
        sections = []

        observation_columns = set(self.get('observation_data_column_names'))
        not_needed = set([self.get('user_id'),
                          self.get('item_id'),
                          self.get('target')])
        num_obs_fields = len(observation_columns.difference(not_needed))

        num_user_features = 0
        if 'user_side_data_column_names' in stats:
            num_user_features = len(self.get('user_side_data_column_names'))
        num_item_features = 0
        if 'item_side_data_column_names' in stats:
            num_item_features = len(self.get('item_side_data_column_names'))

        user_features = self.get('user_side_data_column_names')
        item_features = self.get('item_side_data_column_names')

        section_titles.append("Schema")
        schema_fields = [
            ('User ID', 'user_id'),
            ('Item ID', 'item_id'),
            ('Target', 'target'),
            ('Additional observation features', _precomputed_field(num_obs_fields)),
            ('User side features', _precomputed_field(user_features)),
            ('Item side features', _precomputed_field(item_features))]
        sections.append(schema_fields)

        data_fields = [
            ('Number of observations', 'num_observations'),
            ('Number of users', 'num_users'),
            ('Number of items', 'num_items')]
        section_titles.append("Statistics")
        sections.append(data_fields)

        training_fields = [
            ('Training time', 'training_time')]

        if 'data_load_elapsed_time' in stats:
            training_fields.append(('Data load time',
                                    'data_load_elapsed_time'))
        if 'validation_metrics_elapsed_time' in stats:
            training_fields.append(('Validation metrics time',
                                   'validation_metrics_elapsed_time'))
        section_titles.append("Training summary")
        sections.append(training_fields)

        # Remove any options that should not be shown under "Settings"
        to_ignore = ['random_seed',
                     'user_id',
                     'item_id',
                     'target']

        for k in to_ignore:
            if k in options:
                del options[k]

        def add_ordered_options(name, ordered_options, additional = []):
            option_fields = []

            for k, v in additional:
                option_fields.append((k, _precomputed_field(v)))

            for k in ordered_options:
                if k in options:
                    option_fields.append((k, _precomputed_field(options[k])))
                    del options[k]

            if option_fields:
                section_titles.append(name)
                sections.append(option_fields)

        # Put in a number of things in order, if applicable.

        # Model parameters
        model_parameter_options = [
            "only_top_k",
            "threshold",
            "num_factors",
            "binary_target",
            "side_data_factorization",
            "solver",
            "nmf",
            "max_iterations",
            "similarity_type",
            "training_method"]

        add_ordered_options("Model Parameters", model_parameter_options,
                            [("Model class", self.__class__.__name__)])

        # Regularization type options
        regularization_options = [
            "regularization",
            "regularization_type",
            "linear_regularization",
            "ranking_regularization",
            "unobserved_rating_value",
            "num_sampled_negative_examples",
            "ials_confidence_scaling_type",
            "ials_confidence_scaling_factor"]

        add_ordered_options("Regularization Settings", regularization_options)

        # Optimization stuff
        optimization_settings = [
            "init_random_sigma",
            "sgd_convergence_interval",
            "sgd_convergence_threshold",
            "sgd_max_trial_iterations",
            "sgd_sampling_block_size",
            "sgd_step_adjustment_interval",
            "sgd_step_size",
            "sgd_trial_sample_minimum_size",
            "sgd_trial_sample_proportion",
            "step_size_decrease_rate",
            "additional_iterations_if_unhealthy",
            "adagrad_momentum_weighting",
            "num_tempering_iterations",
            "tempering_regularization_start_value",
            "track_exact_loss"]

        add_ordered_options("Optimization Settings", optimization_settings)

        # clean up
        option_fields = []

        for k, v in _six.iteritems(options):
            option_fields.append((k, _precomputed_field(v)))

        if option_fields:
            section_titles.append("Other Settings")
            sections.append(option_fields)

        return (sections, section_titles)

    def __repr__(self):
        """
        Returns a string description of the model, including (where relevant)
        the schema of the training data, description of the training data,
        training statistics, and model hyperparameters.

        Returns
        -------
        out : string
            A description of the model.
        """
        (sections, section_titles) = self._get_summary_struct()

        return _toolkit_repr_print(self, sections, section_titles, width=32)

    def get_current_options(self):
        """
        A dictionary describing all the parameters of the given model
        and their current setting.
        """
        opts = {'model': self.__proxy__}
        response = _graphlab.toolkits._main.run('recsys_get_current_options', opts)
        return response

    def _set_current_options(self, options):
        """
        Set current options for a model.

        Parameters
        ----------
        options : dict
            A dictionary of the desired option settings. The key should be the name
            of the option and each value is the desired value of the option.
            The possible options are all those returne dy get_default_options().
        """
        opts = self.get_current_options()
        opts.update(options)
        opts['model'] = self.__proxy__
        response = _graphlab.toolkits._main.run('recsys_set_current_options', opts)
        return response

    def __prepare_dataset_parameter(self, dataset):
        """
        Processes the dataset parameter for type correctness.
        Returns it as an SFrame.
        """

        # Translate the dataset argument into the proper type
        if not isinstance(dataset, _SFrame):
            def raise_dataset_type_exception():
                raise TypeError("The dataset parameter must be either an SFrame, "
                                "or a dictionary of (str : list) or (str : value).")

            if type(dataset) is dict:
                if not all(type(k) is str for k in _six.iterkeys(dataset)):
                    raise_dataset_type_exception()

                if all(type(v) in (list, tuple, _array.array) for v in _six.itervalues(dataset)):
                    dataset = _SFrame(dataset)
                else:
                    dataset = _SFrame({k : [v] for k, v in _six.iteritems(dataset)})
            else:
                raise_dataset_type_exception()

        return dataset

    def _get_data_schema(self):
        """
        Returns a dictionary of (column : type) for the data used in the
        model.
        """

        if not hasattr(self, "_data_schema"):

            opts = {'model': self.__proxy__}
            response = _graphlab.toolkits._main.run('recsys_get_data_schema', opts)

            self._data_schema = {k : _graphlab.cython.cy_flexible_type.pytype_from_type_name(v)
                                 for k, v in response["schema"].items()}

        return self._data_schema

    def predict(self, dataset,
                new_observation_data=None, new_user_data=None, new_item_data=None):
        """
        Return a score prediction for the user ids and item ids in the provided
        data set.

        Parameters
        ----------
        dataset : SFrame
            Dataset in the same form used for training.

        new_observation_data : SFrame, optional
            ``new_observation_data`` gives additional observation data
            to the model, which may be used by the models to improve
            score accuracy.  Must be in the same format as the
            observation data passed to ``create``.  How this data is
            used varies by model.

        new_user_data : SFrame, optional
            ``new_user_data`` may give additional user data to the
            model.  If present, scoring is done with reference to this
            new information.  If there is any overlap with the side
            information present at training time, then this new side
            data is preferred.  Must be in the same format as the user
            data passed to ``create``.

        new_item_data : SFrame, optional
            ``new_item_data`` may give additional item data to the
            model.  If present, scoring is done with reference to this
            new information.  If there is any overlap with the side
            information present at training time, then this new side
            data is preferred.  Must be in the same format as the item
            data passed to ``create``.

        Returns
        -------
        out : SArray
            An SArray with predicted scores for each given observation
            predicted by the model.

        See Also
        --------
        recommend, evaluate
        """
        _mt._get_metric_tracker().track('toolkit.recsys.predict')

        if new_observation_data is None:
            new_observation_data = _SFrame()
        if new_user_data is None:
            new_user_data = _SFrame()
        if new_item_data is None:
            new_item_data = _SFrame()


        dataset = self.__prepare_dataset_parameter(dataset)

        def check_type(arg, arg_name, required_type, allowed_types):
            if not isinstance(arg, required_type):
                raise TypeError("Parameter " + arg_name + " must be of type(s) "
                                + (", ".join(allowed_types))
                                + "; Type '" + str(type(arg)) + "' not recognized.")

        check_type(new_observation_data, "new_observation_data", _SFrame, ["SFrame"])
        check_type(new_user_data, "new_user_data", _SFrame, ["SFrame"])
        check_type(new_item_data, "new_item_data", _SFrame, ["SFrame"])

        # Get metadata from C++ object
        opts = {'data_to_predict': dataset,
                'model': self.__proxy__,
               'new_data': new_observation_data,
               'new_user_data': new_user_data,
               'new_item_data': new_item_data
                }

        # Call the C++ function for recommender_model
        response = _graphlab.toolkits._main.run('recsys_predict', opts)
        result = _SFrame(None, _proxy=response['data'])
        return result['prediction']

    def get_similar_items(self, items=None, k=10, verbose=False):
        """
        Get the k most similar items for each item in items.

        Each type of recommender has its own model for the similarity
        between items. For example, the item_similarity_recommender will
        return the most similar items according to the user-chosen
        similarity; the factorization_recommender will return the
        nearest items based on the cosine similarity between latent item
        factors.

        Parameters
        ----------
        items : SArray or list; optional
            An :class:`~graphlab.SArray` or list of item ids for which to get
            similar items. If 'None', then return the `k` most similar items for
            all items in the training set.

        k : int, optional
            The number of similar items for each item.

        verbose : bool, optional
            Progress printing is shown.

        Returns
        -------
        out : SFrame
            A SFrame with the top ranked similar items for each item. The
            columns `item`, 'similar', 'score' and 'rank', where
            `item` matches the item column name specified at training time.
            The 'rank' is between 1 and `k` and 'score' gives the similarity
            score of that item. The value of the score depends on the method
            used for computing item similarities.

        Examples
        --------

        >>> sf = graphlab.SFrame({'user_id': ["0", "0", "0", "1", "1", "2", "2", "2"],
                                  'item_id': ["a", "b", "c", "a", "b", "b", "c", "d"]})
        >>> m = graphlab.item_similarity_recommender.create(sf)
        >>> nn = m.get_similar_items()
        """

        _mt._get_metric_tracker().track('toolkit.recsys.get_similar_items')

        if items is None:
            get_all_items = True
            items = _SArray()
        else:
            get_all_items = False

        if isinstance(items, list):
            items = _SArray(items)

        def check_type(arg, arg_name, required_type, allowed_types):
            if not isinstance(arg, required_type):
                raise TypeError("Parameter " + arg_name + " must be of type(s) "
                                + (", ".join(allowed_types) )
                                + "; Type '" + str(type(arg)) + "' not recognized.")

        check_type(items, "items", _SArray, ["SArray", "list"])
        check_type(k, "k", int, ["int"])

        opt = {'model': self.__proxy__,
               'items': items,
               'get_all_items' : get_all_items,
               'k': k,
               'verbose': verbose}

        response = _graphlab.toolkits._main.run('recsys_get_similar_items', opt)

        neighbors = _SFrame(None, _proxy=response['data'])

        return neighbors

    def get_similar_users(self, users=None, k=10):
        """Get the k most similar users for each entry in `users`.

        Each type of recommender has its own model for the similarity
        between users. For example, the factorization_recommender will
        return the nearest users based on the cosine similarity
        between latent user factors.  (This method is not currently
        available for item_similarity models.)

        Parameters
        ----------
        users : SArray or list; optional
            An :class:`~graphlab.SArray` or list of user ids for which to get
            similar users. If 'None', then return the `k` most similar users for
            all users in the training set.

        k : int, optional
            The number of neighbors to return for each user.

        Returns
        -------
        out : SFrame
            A SFrame with the top ranked similar users for each user. The
            columns `user`, 'similar', 'score' and 'rank', where
            `user` matches the user column name specified at training time.
            The 'rank' is between 1 and `k` and 'score' gives the similarity
            score of that user. The value of the score depends on the method
            used for computing user similarities.

        Examples
        --------

        >>> sf = graphlab.SFrame({'user_id': ["0", "0", "0", "1", "1", "2", "2", "2"],
                                  'item_id': ["a", "b", "c", "a", "b", "b", "c", "d"]})
        >>> m = graphlab.factorization_recommender.create(sf)
        >>> nn = m.get_similar_users()

        """

        _mt._get_metric_tracker().track('toolkit.recsys.get_similar_items')

        if users is None:
            get_all_users = True
            users = _SArray()
        else:
            get_all_users = False

        if isinstance(users, list):
            users = _SArray(users)

        def check_type(arg, arg_name, required_type, allowed_types):
            if not isinstance(arg, required_type):
                raise TypeError("Parameter " + arg_name + " must be of type(s) "
                                + (", ".join(allowed_types) )
                                + "; Type '" + str(type(arg)) + "' not recognized.")

        check_type(users, "users", _SArray, ["SArray", "list"])
        check_type(k, "k", int, ["int"])

        opt = {'model': self.__proxy__,
               'users': users,
               'get_all_users' : get_all_users,
               'k': k}

        response = _graphlab.toolkits._main.run('recsys_get_similar_users', opt)

        neighbors = _SFrame(None, _proxy=response['data'])

        return neighbors


    def recommend(self, users=None, k=10, exclude=None, items=None,
                  new_observation_data=None, new_user_data=None, new_item_data=None,
                  exclude_known=True, diversity=0, random_seed=None,
                  verbose=True):
        """
        Recommend the ``k`` highest scored items for each user.

        Parameters
        ----------
        users :  SArray, SFrame, or list, optional
            Users or observation queries for which to make recommendations.
            For list, SArray, and single-column inputs, this is simply a set
            of user IDs. By default, recommendations are returned for all
            users present when the model was trained. However, if the
            recommender model was created with additional features in the
            ``observation_data`` SFrame, then a corresponding SFrame of
            observation queries -- observation data without item or target
            columns -- can be passed to this method. For example, a model
            trained with user ID, item ID, time, and rating columns may be
            queried using an SFrame with user ID and time columns.  In this
            case, the user ID column must be present, and all column names
            should match those in the ``observation_data`` SFrame passed to
            ``create.``

        k : int, optional
            The number of recommendations to generate for each user.

        items : SArray, SFrame, or list, optional
            Restricts the items from which recommendations can be made. If
            ``items`` is an SArray, list, or SFrame with a single column,
            only items from the given set will be recommended. This can be
            used, for example, to restrict the recommendations to items
            within a particular category or genre. If ``items`` is an
            SFrame with user ID and item ID columns, then the item
            restriction is specialized to each user. For example, if
            ``items`` contains 3 rows with user U1 -- (U1, I1), (U1, I2),
            and (U1, I3) -- then the recommendations for user U1 are
            chosen from items I1, I2, and I3. By default, recommendations
            are made from all items present when the model was trained.

        new_observation_data : SFrame, optional
            ``new_observation_data`` gives additional observation data
            to the model, which may be used by the models to improve
            score and recommendation accuracy.  Must be in the same
            format as the observation data passed to ``create``.  How
            this data is used varies by model.

        new_user_data : SFrame, optional
            ``new_user_data`` may give additional user data to the
            model.  If present, scoring is done with reference to this
            new information.  If there is any overlap with the side
            information present at training time, then this new side
            data is preferred.  Must be in the same format as the user
            data passed to ``create``.

        new_item_data : SFrame, optional
            ``new_item_data`` may give additional item data to the
            model.  If present, scoring is done with reference to this
            new information.  If there is any overlap with the side
            information present at training time, then this new side
            data is preferred.  Must be in the same format as the item
            data passed to ``create``.

        exclude : SFrame, optional
            An :class:`~graphlab.SFrame` of user / item pairs.  The
            column names must be equal to the user and item columns of
            the main data, and it provides the model with user/item
            pairs to exclude from the recommendations.  These
            user-item-pairs are always excluded from the predictions,
            even if exclude_known is False.

        exclude_known : bool, optional
            By default, all user-item interactions previously seen in
            the training data, or in any new data provided using
            new_observation_data.., are excluded from the
            recommendations.  Passing in ``exclude_known = False``
            overrides this behavior.

        diversity : non-negative float, optional
            If given, then the recommend function attempts chooses a set
            of `k` items that are both highly scored and different from
            other items in that set. It does this by first retrieving
            ``k*(1+diversity)`` recommended items, then randomly
            choosing a diverse set from these items.  Suggested values
            for diversity are between 1 and 3.

        random_seed : int, optional
            If diversity is larger than 0, then some randomness is used;
            this controls the random seed to use for randomization.  If
            None, will be different each time.

        verbose : bool, optional
            If True, print the progress of generating recommendation.

        Returns
        -------
        out : SFrame
            A SFrame with the top ranked items for each user. The
            columns are: ``user_id``, ``item_id``, *score*,
            and *rank*, where ``user_id`` and ``item_id``
            match the user and item column names specified at training
            time.  The rank column is between 1 and ``k`` and gives
            the relative score of that item.  The value of score
            depends on the method used for recommendations.

        See Also
        --------
        recommend_from_interactions
        predict
        evaluate
        """
        _mt._get_metric_tracker().track('toolkit.recsys.recommend')
        assert type(k) == int

        column_types = self._get_data_schema()
        user_id = self.get('user_id')
        item_id = self.get('item_id')
        user_type = column_types[user_id]
        item_type = column_types[item_id]

        __null_sframe = _SFrame()

        if users is None:
            users = __null_sframe
        if exclude is None:
            exclude = __null_sframe
        if items is None:
            items = __null_sframe
        if new_observation_data is None:
            new_observation_data = __null_sframe
        if new_user_data is None:
            new_user_data = __null_sframe
        if new_item_data is None:
            new_item_data = __null_sframe

        if isinstance(users, list) or (_HAS_NUMPY and isinstance(users, _numpy.ndarray)):
            users = _SArray(users)

            # allow to take a list of dictionaries of the form [{'user_id':1,'time':10}] etc.
            if users.dtype() == dict:
                users = users.unpack(column_name_prefix='')

        if isinstance(users, _SArray):
            users = _SFrame({user_id: users})

        if isinstance(items, list) or (_HAS_NUMPY and isinstance(items, _numpy.ndarray)):
            items = _SArray(items, dtype = item_type)

        if isinstance(items, _SArray):
            items = _SFrame({item_id: items})

        # Check type of incoming data.
        def check_type(arg, arg_name, required_type, allowed_types):
            if not isinstance(arg, required_type):
                raise TypeError("Parameter " + arg_name + " must be of type(s) "
                                + (", ".join(allowed_types))
                                + "; Type '" + str(type(arg)) + "' not recognized.")

        check_type(users, "users", _SFrame, ["SArray", "list", "SFrame", "numpy.ndarray"])
        check_type(exclude, "exclude", _SFrame, ["SFrame"])
        check_type(items, "items", _SFrame, ["SFrame", "SArray", "list", "numpy.ndarray"])
        check_type(new_observation_data, "new_observation_data", _SFrame, ["SFrame"])
        check_type(new_user_data, "new_user_data", _SFrame, ["SFrame"])
        check_type(new_item_data, "new_item_data", _SFrame, ["SFrame"])

        # See if we are in the situation where there are no users
        # originally.  In this case, the default type of the user
        # column is string, so we have to be mindful of that when
        # making recommendations and possibly cast it to string if
        # needed.

        # The only case where we need to deal with the user id is when
        # it's used to link with rated items in new_observation_data,
        # thus handle that case explicitly and error out in others.

        cast_user_to_string_type = False

        if self.get("num_users") == 0:
            cast_user_to_string_type = True

            if users.num_rows() != 0:
                # In this case, the user column has actually been set to a
                # string type, so we need to make sure that we cast
                # everything back and forth to that to preserve type.

                if new_observation_data.num_rows() == 0:
                    raise ValueError("When users are not specified with the model, "
                                     "new_observation_data must be set in order to make recommendations.")

                new_observation_data[user_id] = new_observation_data[user_id].astype(user_type)

            else:
                print("WARNING: No users specified to model at creation time, so "
                      "calling recommend() for all users returns empty SFrame.")

        # Cast to the appropriate type if necessary.
        if users.num_rows() != 0:
            try:
                user_column = users[user_id]
            except RuntimeError:
                raise _ToolkitError("User column '%s' not present in input user data." % user_id)

            if cast_user_to_string_type:
                assert new_observation_data.num_rows() != 0
                original_user_type = user_column.dtype()
                users[user_id] = user_column.astype(str)

            elif user_column.dtype() != user_type:
                users[user_id] = user_column.astype(user_type)

        try:
            diversity = float(diversity)
        except Exception:
            raise TypeError("Parameter diversity must be a floating point value equal to or larger than 0.")

        if diversity < 0:
            raise TypeError("Parameter diversity must be a floating point value equal to or larger than 0.")

        if random_seed is None:
            random_seed = hash("%.20f" % _time.time())
        else:
            try:
                random_seed = int(random_seed)
            except TypeError:
                raise TypeError("random_seed must be integer.")

        opt = {'model': self.__proxy__,
               'query': users,
               'top_k': k,
               'exclude': exclude,
               'restrictions': items,
               'new_data': new_observation_data,
               'new_user_data': new_user_data,
               'new_item_data': new_item_data,
               'exclude_known': exclude_known,
               'diversity' : diversity,
               'random_seed' : random_seed
               }
        response = _graphlab.toolkits._main.run('recsys_recommend', opt, verbose=verbose)
        recs = _SFrame(None, _proxy=response['data'])

        if cast_user_to_string_type:
            recs[user_id] = recs[user_id].astype(original_user_type)

        return recs

    def recommend_from_interactions(
            self, observed_items, k=10, exclude=None, items=None,
            new_user_data=None, new_item_data=None,
            exclude_known=True, diversity=0, random_seed=None,
            verbose=True):
        """
        Recommend the ``k`` highest scored items based on the
        interactions given in `observed_items.`

        Parameters
        ----------
        observed_items : SArray, SFrame, or list
            A list/SArray of items to use to make recommendations, or
            an SFrame of items and optionally ratings and/or other
            interaction data.  The model will then recommend the most
            similar items to those given.  If ``observed_items`` has a user
            column, then it must be only one user, and the additional
            interaction data stored in the model is also used to make
            recommendations.

        k : int, optional
            The number of recommendations to generate.

        items : SArray, SFrame, or list, optional
            Restricts the items from which recommendations can be
            made. ``items`` must be an SArray, list, or SFrame with a
            single column containing items, and all recommendations
            will be made from this pool of items.  This can be used,
            for example, to restrict the recommendations to items
            within a particular category or genre. By default,
            recommendations are made from all items present when the
            model was trained.

        new_user_data : SFrame, optional
            ``new_user_data`` may give additional user data to the
            model.  If present, scoring is done with reference to this
            new information.  If there is any overlap with the side
            information present at training time, then this new side
            data is preferred.  Must be in the same format as the user
            data passed to ``create``.

        new_item_data : SFrame, optional
            ``new_item_data`` may give additional item data to the
            model.  If present, scoring is done with reference to this
            new information.  If there is any overlap with the side
            information present at training time, then this new side
            data is preferred.  Must be in the same format as the item
            data passed to ``create``.

        exclude : SFrame, optional
            An :class:`~graphlab.SFrame` of items or user / item
            pairs.  The column names must be equal to the user and
            item columns of the main data, and it provides the model
            with user/item pairs to exclude from the recommendations.
            These user-item-pairs are always excluded from the
            predictions, even if exclude_known is False.

        exclude_known : bool, optional
            By default, all user-item interactions previously seen in
            the training data, or in any new data provided using
            new_observation_data.., are excluded from the
            recommendations.  Passing in ``exclude_known = False``
            overrides this behavior.

        diversity : non-negative float, optional
            If given, then the recommend function attempts chooses a set
            of `k` items that are both highly scored and different from
            other items in that set. It does this by first retrieving
            ``k*(1+diversity)`` recommended items, then randomly
            choosing a diverse set from these items.  Suggested values
            for diversity are between 1 and 3.

        random_seed : int, optional
            If diversity is larger than 0, then some randomness is used;
            this controls the random seed to use for randomization.  If
            None, then it will be different each time.

        verbose : bool, optional
            If True, print the progress of generating recommendation.

        Returns
        -------
        out : SFrame
            A SFrame with the top ranked items for each user. The
            columns are: ``item_id``, *score*, and *rank*, where
            ``user_id`` and ``item_id`` match the user and item column
            names specified at training time.  The rank column is
            between 1 and ``k`` and gives the relative score of that
            item.  The value of score depends on the method used for
            recommendations.

        observed_items: list, SArray, or SFrame

        """

        column_types = self._get_data_schema()
        user_id = self.get("user_id")
        item_id = self.get("item_id")
        user_type = column_types[user_id]
        item_type = column_types[item_id]

        if not hasattr(self, "_implicit_user_name"):
            import hashlib
            import time
            self._implicit_user_name = None #("implicit-user-%s"
#                                        % hashlib.md5("%0.20f" % time.time()).hexdigest()[:12])

        if isinstance(observed_items, list):
            observed_items = _SArray(observed_items, dtype = item_type)
        if isinstance(observed_items, _SArray):
            observed_items = _SFrame({self.get("item_id") : observed_items})

        if not isinstance(observed_items, _SFrame):
            raise TypeError("observed_items must be a list or SArray of items, or an SFrame of items "
                            "and optionally ratings or other interaction information.")

        # Don't modify the user's argument (if it's an SFrame).
        observed_items = observed_items.copy()

        # If a user id is present, then use that as the query user id
        # (making sure there is only one present).  If not, then use
        # the local fake user id.
        if user_id in observed_items.column_names():
            main_user_value = observed_items[user_id][0]
            if (observed_items[user_id] != main_user_value).any():
                raise ValueError("To recommend items for more than one user, use `recommend()` and "
                                 "supply new interactions using new_observation_data.")
            users = _SArray([main_user_value], dtype = user_type)

        else:
            users = _SArray([self._implicit_user_name], dtype = user_type)
            observed_items[user_id] = self._implicit_user_name

        if observed_items[user_id].dtype() != user_type:
            observed_items[user_id] = observed_items[user_id].astype(user_type)

        # Check the rest of the arguments.
        if exclude is not None:
            if isinstance(exclude, list):
                exclude = _SArray(exclude, dtype = item_type)
            if isinstance(exclude, _SArray):
                exclude = _SFrame({item_id : exclude})
            if user_id not in exclude.column_names():
                exclude[user_id] = self._implicit_user_name
                exclude[user_id] = exclude[user_id].astype(user_type)

        recommendations = self.recommend(
            users                = users,
            new_observation_data = observed_items,
            k                    = k,
            items                = items,
            new_user_data        = new_user_data,
            new_item_data        = new_item_data,
            exclude_known        = exclude_known,
            diversity            = diversity,
            random_seed          = random_seed,
            verbose              = verbose)

        del recommendations[user_id]

        return recommendations


    def _training_stats(self):
        """
        Get information about model creation, e.g. time elapsed during
        model fitting, data loading, and more.

        Note: This method will be *deprecated* soon. Please use m.summary()
        instead.

        Returns
        -------
        out : dict
            Statistics about model training, e.g. runtime.

        """
        _logging.warning("This method will be deprecated soon. Please use m.summary().")
        _mt._get_metric_tracker().track('toolkit.recsys._training_stats')

        opts = {'model': self.__proxy__}
        response = _graphlab.toolkits._main.run("recsys_get_train_stats", opts)
        return response

    def evaluate_precision_recall(self, dataset, cutoffs=list(range(1,11,1))+list(range(11,50,5)),
                                  skip_set=None, exclude_known=True,
                                  verbose=True, **kwargs):
        """
        Compute a model's precision and recall scores for a particular dataset.

        Parameters
        ----------
        dataset : SFrame
            An SFrame in the same format as the one used during training.
            This will be compared to the model's recommendations, which exclude
            the (user, item) pairs seen at training time.

        cutoffs : list, optional
            A list of cutoff values for which one wants to evaluate precision
            and recall, i.e. the value of k in "precision at k".

        skip_set : SFrame, optional
            Passed to :meth:`recommend` as ``exclude``.

        exclude_known : bool, optional
            Passed to :meth:`recommend` as ``exclude_known``. If True, exclude
            training item from recommendation.

        verbose : bool, optional
            Enables verbose output. Default is verbose.

        **kwargs
            Additional keyword arguments are passed to the recommend
            function, whose returned recommendations are used for evaluating
            precision and recall of the model.

        Returns
        -------
        out : dict
            Contains the precision and recall at each cutoff value and each
            user in ``dataset``.

        Examples
        --------

        >>> import graphlab as gl
        >>> sf = gl.SFrame('https://static.turi.com/datasets/audioscrobbler')
        >>> train, test = gl.recommender.util.random_split_by_user(sf)
        >>> m = gl.recommender.create(train)
        >>> m.evaluate_precision_recall(test)

        See Also
        --------
        graphlab.recommender.util.precision_recall_by_user
        """

        _mt._get_metric_tracker().track('toolkit.recsys.evaluate_precision_recall')

        user_column = self.get('user_id')
        item_column = self.get('item_id')
        assert user_column in dataset.column_names() and \
               item_column in dataset.column_names(), \
            'Provided data set must have a column pertaining to user ids and \
             item ids, similar to what we had during training.'

        dataset = self.__prepare_dataset_parameter(dataset)

        users = dataset[self.get('user_id')].unique()

        dataset = dataset[[self.get('user_id'), self.get('item_id')]]

        recs = self.recommend(users=users, k=max(cutoffs), exclude=skip_set,
                              exclude_known=exclude_known,
                              verbose=verbose,
                              **kwargs)

        precision_recall_by_user = _graphlab.recommender.util.precision_recall_by_user(dataset, recs, cutoffs)

        ret = {'precision_recall_by_user': precision_recall_by_user}

        pr_agg = precision_recall_by_user.groupby(
             'cutoff',
             operations={'precision' : _Aggregate.MEAN('precision'),
                         'recall'    : _Aggregate.MEAN('recall')})

        pr_agg = pr_agg[['cutoff', 'precision', 'recall']]
        ret["precision_recall_overall"] = pr_agg.sort("cutoff")

        return ret

    def evaluate_rmse(self, dataset, target):
        """
        Evaluate the prediction error for each user-item pair in the given data
        set.

        Parameters
        ----------
        dataset : SFrame
            An SFrame in the same format as the one used during training.

        target : str
            The name of the target rating column in `dataset`.

        Returns
        -------
        out : dict
            A dictionary with three items: 'rmse_by_user' and 'rmse_by_item',
            which are SFrames containing the average rmse for each user and
            item, respectively; and 'rmse_overall', which is a float.

        Examples
        --------
        >>> import graphlab as gl
        >>> sf = gl.SFrame('https://static.turi.com/datasets/audioscrobbler')
        >>> train, test = gl.recommender.util.random_split_by_user(sf)
        >>> m = gl.recommender.create(train, target='target')
        >>> m.evaluate_rmse(test, target='target')

        See Also
        --------
        graphlab.evaluation.rmse
        """

        _mt._get_metric_tracker().track('toolkit.recsys.evaluate_rmse')

        assert target in dataset.column_names(), \
               'Provided dataset must contain a target column with the same \
                name as the target used during training.'
        y = dataset[target]
        yhat = self.predict(dataset)
        user_column = self.get('user_id')
        item_column = self.get('item_id')
        assert user_column in dataset.column_names() and \
               item_column in dataset.column_names(), \
            'Provided data set must have a column pertaining to user ids and \
             item ids, similar to what we had during training.'

        result = dataset[[user_column, item_column]]
        result['sq_error'] = (y - yhat) * (y - yhat)
        rmse_by_user = result.groupby(user_column,
                {'rmse':_graphlab.aggregate.AVG('sq_error'),
                 'count':_graphlab.aggregate.COUNT})
        rmse_by_user['rmse'] = rmse_by_user['rmse'].apply(lambda x: x**.5)
        rmse_by_item = result.groupby(item_column,
                {'rmse':_graphlab.aggregate.AVG('sq_error'),
                 'count':_graphlab.aggregate.COUNT})
        rmse_by_item['rmse'] = rmse_by_item['rmse'].apply(lambda x: x**.5)
        overall_rmse = result['sq_error'].mean() ** .5

        return {'rmse_by_user': rmse_by_user,
                'rmse_by_item': rmse_by_item,
                'rmse_overall': overall_rmse}

    @_collect_model_workflow
    def evaluate(self, dataset, metric='auto',
                 exclude_known_for_precision_recall=True,
                 target=None,
                 verbose=True, **kwargs):
        r"""
        Evaluate the model's ability to make rating predictions or
        recommendations.

        If the model is trained to predict a particular target, the
        default metric used for model comparison is root-mean-squared error
        (RMSE). Suppose :math:`y` and :math:`\widehat{y}` are vectors of length
        :math:`N`, where :math:`y` contains the actual ratings and
        :math:`\widehat{y}` the predicted ratings. Then the RMSE is defined as

        .. math::

            RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (\widehat{y}_i - y_i)^2} .

        If the model was not trained on a target column, the default metrics for
        model comparison are precision and recall. Let
        :math:`p_k` be a vector of the :math:`k` highest ranked recommendations
        for a particular user, and let :math:`a` be the set of items for that
        user in the groundtruth `dataset`. The "precision at cutoff k" is
        defined as

        .. math:: P(k) = \frac{ | a \cap p_k | }{k}

        while "recall at cutoff k" is defined as

        .. math:: R(k) = \frac{ | a \cap p_k | }{|a|}

        Parameters
        ----------
        dataset : SFrame
            An SFrame that is in the same format as provided for training.

        metric : str, {'auto', 'rmse', 'precision_recall'}, optional
            Metric to use for evaluation. The default automatically chooses
            'rmse' for models trained with a `target`, and 'precision_recall'
            otherwise.

        exclude_known_for_precision_recall : bool, optional
            A useful option for evaluating precision-recall. Recommender models
            have the option to exclude items seen in the training data from the
            final recommendation list. Set this option to True when evaluating
            on test data, and False when evaluating precision-recall on training
            data.

        target : str, optional
            The name of the target column for evaluating rmse. If the model is
            trained with a target column, the default is to using the same
            column. If the model is trained without a target column and `metric`
            is set to 'rmse', this option must provided by user.

        verbose : bool, optional
            Enables verbose output. Default is verbose.

        **kwargs
            When `metric` is set to 'precision_recall', these parameters
            are passed on to :meth:`evaluate_precision_recall`.

        Returns
        -------
        out : SFrame or dict
            Results from the model evaluation procedure. If the model is trained
            on a target (i.e. RMSE is the evaluation criterion), a dictionary
            with three items is returned: items *rmse_by_user* and
            *rmse_by_item* are SFrames with per-user and per-item RMSE, while
            *rmse_overall* is the overall RMSE (a float). If the model is
            trained without a target (i.e. precision and recall are the
            evaluation criteria) an :py:class:`~graphlab.SFrame` is returned
            with both of these metrics for each user at several cutoff values.

        Examples
        --------
        >>> import graphlab as gl
        >>> sf = gl.SFrame('https://static.turi.com/datasets/audioscrobbler')
        >>> train, test = gl.recommender.util.random_split_by_user(sf)
        >>> m = gl.recommender.create(train, target='target')
        >>> eval = m.evaluate(test)

        See Also
        --------
        evaluate_precision_recall, evaluate_rmse, precision_recall_by_user
        """

        _mt._get_metric_tracker().track('toolkit.recsys.evaluate')

        ret = {}

        dataset = self.__prepare_dataset_parameter(dataset)

        # If the model does not have a target column, compute prec-recall.
        if metric in ['precision_recall', 'auto']:
            results = self.evaluate_precision_recall(dataset,
                                                     exclude_known=exclude_known_for_precision_recall,
                                                     verbose=verbose,
                                                     **kwargs)
            ret.update(results)
            if verbose:
                print("\nPrecision and recall summary statistics by cutoff")
                print(results['precision_recall_by_user'].groupby('cutoff', \
                        {'mean_precision': _graphlab.aggregate.AVG('precision'),
                         'mean_recall': _graphlab.aggregate.AVG('recall')}).topk('cutoff', reverse=True))
        if metric in ['rmse', 'auto']:
            if target is None:
                target = self.get('target')
            if target is None or target == "":
                _logging.warning("Model trained without a target. Skipping RMSE computation.")
            else:
                results = self.evaluate_rmse(dataset, target)
                ret.update(results)

                if verbose:
                    print("\nOverall RMSE: ", results['rmse_overall'])
                    print("\nPer User RMSE (best)")
                    print(results['rmse_by_user'].topk('rmse', 1, reverse=True))
                    print("\nPer User RMSE (worst)")
                    print(results['rmse_by_user'].topk('rmse', 1))
                    print("\nPer Item RMSE (best)")
                    print(results['rmse_by_item'].topk('rmse', 1, reverse=True))
                    print("\nPer Item RMSE (worst)")
                    print(results['rmse_by_item'].topk('rmse', 1))
        if metric not in ['rmse', 'precision_recall', 'auto']:
            raise ValueError('Unknown evaluation metric %s, supported metrics are [\"rmse\", \"precision_recall\"]' % metric)

        return ret

    def show(self, view=None, model_type='recommender'):
        """
        show(view=None)
        Visualize a model with GraphLab Create :mod:`~graphlab.canvas`. This function starts Canvas
        if it is not already running. If the Model has already been plotted,
        this function will update the plot.

        Parameters
        ----------
        view : str, optional
            The name of the Model view to show. Can be one of:

            - Summary: Shows the statistics of the training process such as size of the data and time cost. The summary also shows the parameters and settings for the model training process if available.
            - Evaluation: Shows precision recall plot as line chart. Tooltip is provided for pointwise analysis. Precision recall values are shown in the tooltip at any given cutoff value the mouse points to.
            - Comparison: Shows the precision recall metrics for multiple models. It also support tooltip and highlight (mouse over) and focus (mouse selection on the legend) for inspection in detail.

        Returns
        -------
        view : graphlab.canvas.view.View
            An object representing the GraphLab Canvas view.

        See Also
        --------
        canvas

        Examples
        --------
        Suppose 'm' is a Model, we can view it in GraphLab Canvas using:

        >>> m.show()
        """
        return super(_Recommender, self).show(view=view, model_type=model_type)

    def _get_popularity_baseline(self):

        """
        Returns a new popularity model matching the data set this model was
        trained with.  Can be used for comparison purposes.
        """

        opts = {'model': self.__proxy__}
        response = _graphlab.toolkits._main.run('recsys_get_popularity_baseline', opts)


        from .popularity_recommender import PopularityRecommender

        return PopularityRecommender(response["popularity_model"])

    def _get_item_intersection_info(self, item_pairs):
        """
        For a collection of item -> item pairs, returns information about the
        users in that intersection.

        Parameters
        ----------

        item_pairs : 2-column SFrame of two item columns, or a list of
           (item_1, item_2) tuples.

        Returns
        -------
        out : SFrame
           A SFrame with the two item columns given above, the number of
           users that rated each, and a dictionary mapping the user to a
           pair of the ratings, with the first rating being the rating of
           the first item and the second being the rating of the second item.
           If no ratings are provided, these values are always 1.0.
        """

        if type(item_pairs) is list:
            if not all(type(t) in [list, tuple] and len(t) == 2 for t in item_pairs):
                raise TypeError("item_pairs must be 2-column SFrame of two item "
                                "columns, or a list of (item_1, item_2) tuples. ")

            item_name = self.get('item_id')
            item_pairs = _graphlab.SFrame({item_name + "_1" : [v1 for v1, v2 in item_pairs],
                                           item_name + "_2" : [v2 for v1, v2 in item_pairs]})

        if not isinstance(item_pairs, _graphlab.SFrame):
            raise TypeError("item_pairs must be 2-column SFrame of two item "
                            "columns, or a list of (item_1, item_2) tuples. ")


        opts = {'model': self.__proxy__,
                'item_pairs' : item_pairs}

        response = _graphlab.toolkits._main.run('recsys_get_item_intersection_info', opts)
        return _SFrame(None, _proxy=response['item_intersections'])
