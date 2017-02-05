'''
Tools for fitting models using a set of parameter values.

The :py:func:`create` method provides the ability to do
parameter search for a variety of supported GraphLab Create and sklearn models.
Initial suggestions of parameter ranges are provided. Any search parameters
that are provided by the user will be used in place of a set of default
parameter ranges.

If you want to perform a search over only the provided parameters, please see
the :py:func:`grid_search.create` and :py:func:`random_search.create`
methods. If you alread have a list of parameter settings to use,
please see the :py:func:`manual_search.create` method.
'''

__all__ = ['grid_search', 'random_search', 'manual_search']

from . import manual_search
from . import random_search
from . import grid_search
from ._model_parameter_search_evaluators import default_evaluator as _default_evaluator
from ._defaults import _sensible_defaults
from ._model_parameter_search import ModelSearchJob, _add_docstring, _DATA_DOCSTRING, _MODEL_FACTORY_DOCSTRING, _MODEL_PARAMETERS_DOCSTRING, _EVALUATOR_DOCSTRING, _ENVIRONMENT_DOCSTRING, _RETURN_MODEL_DOCSTRING, _RETURNS_DOCSTRING, _PERFORM_TRIAL_RUN_DOCSTRING


_EXAMPLES_DOCSTRING = '''
    Examples
    --------

    .. sourcecode:: python

        >>> import graphlab
        >>> from graphlab import model_parameter_search
        >>> from graphlab import SFrame

        >>> regression_data = SFrame({'x': range(100), 'y': [0,1] * (50)})
        >>> training, validation = regression_data.random_split(0.8)
        >>> params = {'target': 'y'}

        # Search over a grid of multiple hyper parameters, with validation set
        >>> job = model_parameter_search.create((training, validation),
                                                graphlab.linear_regression.create,
                                                params, max_models=4)
        >>> results = job.get_results()
        >>> results.column_names()
        ['model_id',
         'l1_penalty',
         'l2_penalty',
         'step_size',
         'target',
         'training_rmse',
         'validation_rmse']

        >>> results[['step_size', 'validation_rmse']]
        +-----------+-----------------+
        | step_size | validation_rmse |
        +-----------+-----------------+
        |    0.01   |  0.500233862602 |
        |    0.01   |  0.500015657709 |
        |    0.01   |  0.501531488066 |
        |    0.01   |  0.500015657709 |
        |   0.001   |  0.496787248633 |
        |    0.01   |  0.498433727407 |
        |   0.001   |  0.487540155119 |
        |    0.01   |  0.500015657709 |
        |   0.001   |  0.505572888694 |
        |   0.001   |  0.490306163748 |
        +-----------+-----------------+

        # Each model corresponds one row in the summary sframe.
        >>> len(job.get_models())
        4

        >>> model_with_first_parameter_set = job.get_models()[0]

    The job can also be executed easily on EC2 as follows:

    .. sourcecode:: python

        >>> env = graphlab.deploy.environment.EC2(...)
        >>> job = model_parameter_search.create((training, validation),
                                                graphlab.linear_regression.create,
                                                params,
                                                environment=env)

    Default evaluator works for recommender, classifier, regression
    and topic models by using the default evaluation metrics respectively
    on the training_set and validation set. The regression models summary
    contains RMSE as shown above.

    .. sourcecode:: python

        # Classifier uses classification accuracy
        >>> params = {'target': 'y'}
        >>> job = model_parameter_search.create((training, validation),
                                                graphlab.svm_classifier.create,
                                                params)
        >>> job.get_results()
        +----------+---------+-------------------+---------------------+
        | model_id | penalty | training_accuracy | validation_accuracy |
        +----------+---------+-------------------+---------------------+
        |    0     |  0.001  |   0.480519480519  |    0.565217391304   |
        |    1     |   0.01  |   0.519480519481  |    0.434782608696   |
        +----------+---------+-------------------+---------------------+
        [2 rows x 4 columns]

        # Recommender uses precision/recall and rmse (if applicable)
        >>> recommender_data = SFrame({'user': range(10),
                                       'item': range(10),
                                       'rating': [1, 2] * 5})
        >>> params = {'user_id': 'user', 'item_id': 'item', 'target': 'rating',
                      'num_factors': [1, 5]}

        >>> job = model_parameter_search.create(recommender_data,
                                                graphlab.factorization_recommender.create,
                                                params)
        >>> job.get_results()
        +----------+-------------+----------------------+-------------------+
        | model_id | num_factors | training_precision@5 | training_recall@5 |
        +----------+-------------+----------------------+-------------------+
        |    0     |      1      |         0.1          |        0.5        |
        |    1     |      5      |         0.1          |        0.5        |
        +----------+-------------+----------------------+-------------------+
        +-------------------+
        |   training_rmse   |
        +-------------------+
        | 0.000503940148226 |
        | 0.000503941208739 |
        +-------------------+
        [2 rows x 5 columns]

        # Topic model uses perplexity
        >>> text_data = SFrame({'bag_of_words': [{'fish': 10, 'chips':20}] * 10})
        >>> job = model_parameter_search.create((text_data, text_data),
                                                graphlab.topic_model.create,
                                                {'num_topics': [2, 5]})
        >>> job.get_results()
        +----------+------------+---------------+
        | model_id | num_topics |   perplexity  |
        +----------+------------+---------------+
        |    0     |     2      | 1.91656870839 |
        |    1     |     5      | 1.99884277248 |
        +----------+------------+---------------+
        [2 rows x 3 columns]

    Other models does not have default evaluator, and the summary will contain
    a None valued column. For example, for Kmeans model, we can
    do the following.

    .. sourcecode:: python

        >>> clustering_data = SFrame({'x': range(10), 'y': range(10)})
        >>> job = model_parameter_search.create(clustering_data,
                                                graphlab.kmeans.create,
                                                {'num_clusters': [2, 5]})
        >>> job.get_results()
        +----------+--------------+--------+
        | model_id | num_clusters | metric |
        +----------+--------------+--------+
        |    0     |      2       |  None  |
        |    1     |      5       |  None  |
        +----------+--------------+--------+
        [2 rows x 3 columns]


    You can define your own evaluator function by defining
    a function which takes model, training_data, validation_data
    as input, and output a dictionary with metric_name as key
    and metric_value as value.

    For example, for the above KMeans clustering model, use an
    evaluator which output the average within-cluster distance.

    .. sourcecode:: python

        >>> def custom_evaluator(model, train, test):
                mean_squared_distance = model['cluster_info']['sum_squared_distance'].mean()
                return {'mean_distance': mean_squared_distance}
        >>> params = {'num_clusters': [2, 5]}
        >>> job = model_parameter_search.create(clustering_data,
                                                graphlab.kmeans.create,
                                                params,
                                                evaluator=custom_evaluator)
        >>> job.get_results()
        +----------+--------------+---------------+
        | model_id | num_clusters | mean_distance |
        +----------+--------------+---------------+
        |    0     |      2       | 8.48528137424 |
        |    1     |      5       | 1.41421356237 |
        +----------+--------------+---------------+
        [2 rows x 3 columns]

    You can do model parameter search on user-defined functions. The
    function must take a training_set and keyword arguments dictionary
    as input, and output a function that returns some model object. This model
    must be valid input for the function passed as the "evaluator" parameter.

    For example, the following code defines a function that takes a training
    set and returns a function that can provide a score givin a dataset.
    The evaluation function simply reports this score as the metric.
    This is done for each provided value of K. In the example the model
    simply counts the number of rows in the dataset.

    .. sourcecode:: python

        >>> def custom_create(training_set, **params):
                def predict(x):
                    return x.num_rows()
                return predict
        >>> def custom_eval(model, training_set, validation_set):
                return {'training_nrows'  : model(training_set),
                        'validation_nrows': model(validation_set)}

        >>> params = {'K': [1,3,5]}
        >>> job = gl.model_parameter_search.create((training, validation),
                                                  custom_create,
                                                  params,
                                                  evaluator=custom_eval)

    You can also perform model parameter search on scikit-learn model objects.
    For the training_set simply pass a tuple containing the training features
    and target values, respectively. Each execution will call `fit` on a
    constructed model using the provided training data. The default evaluator
    will call `score` on the model with the `training_set` (and the
    `validation_set` if provided). Note that the training set must be a
    tuple of numpy matrixs having dimension N x p and N x 1, respectively,
    where N is the number of instances and p is the number of features.

    .. sourcecode:: python

        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> y = data['target']
        >>> X = data['data']
        >>> params = {'max_depth': [3, 4, 5]}
        >>> train = (X, y)
        >>> valid = (X, y)

        >>> job = gl.model_parameter_search.create((train, valid),
                                                   GradientBoostingClassifier,
                                                   params)
'''


@_add_docstring(param_data=_DATA_DOCSTRING,
               param_model_factory=_MODEL_FACTORY_DOCSTRING,
               param_model_params=_MODEL_PARAMETERS_DOCSTRING,
               param_evaluator=_EVALUATOR_DOCSTRING,
               param_environment=_ENVIRONMENT_DOCSTRING,
               param_return_model=_RETURN_MODEL_DOCSTRING,
               param_perform_trial_run=_PERFORM_TRIAL_RUN_DOCSTRING,
               param_returns=_RETURNS_DOCSTRING,
               examples=_EXAMPLES_DOCSTRING)
def create(datasets,
           model_factory,
           model_parameters,
           evaluator=_default_evaluator,
           environment=None,
           perform_trial_run=True,
           return_model=True,
           max_models=10):
    """
    Evaluate model performance, in parallel, over a set of parameters.

    Parameters
    ----------
    {param_data}
    {param_model_factory}
    {param_model_params}
        Any search parameters that are provided will be used in place of
        a set of default parameter ranges for a set of supported models.
        If you want to perform a search over **only** the provided
        parameters, please see the
        :py:func:`grid_search.create` and
        :py:func:`random_search.create` modules.

    {param_evaluator}
    {param_environment}
    {param_perform_trial_run}
    {param_return_model}
    {param_returns}

    See Also
    --------
    random_search.create, grid_search.create, manual_search.create

    {examples}

    """

    model_params = _sensible_defaults(model_factory, datasets)
    model_params.update(model_parameters)

    return random_search.create(datasets,
                                model_factory,
                                model_params,
                                evaluator=evaluator,
                                environment=environment,
                                return_model=return_model,
                                perform_trial_run=perform_trial_run,
                                max_models=max_models)
