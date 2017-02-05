from ._model_parameter_search_evaluators import default_evaluator as _default_evaluator
import random
from ._model_parameter_search import _create_model_search
from ._model_parameter_search import _add_docstring, _DATA_DOCSTRING, _MODEL_FACTORY_DOCSTRING, _MODEL_PARAMETERS_DOCSTRING, _EVALUATOR_DOCSTRING, _ENVIRONMENT_DOCSTRING, _RETURN_MODEL_DOCSTRING, _MAX_MODELS_DOCSTRING, _RETURNS_DOCSTRING, _PERFORM_TRIAL_RUN_DOCSTRING


def _random_choice(model_parameters):
    """
    For a given dictionary of model parameters, this method samples a random
    parameter value for each key in the dictionary.

    Parameters
    ----------

    model_parameters : dict
      The behavior for each value depends on the following rules:

    * list : picks random element from the list
    * iterable: pick an element using random.choice
    * scipy.distribution: calls v.rvs(1) to sample a single value from the distribution
    * int, float, str: uses this value.

    Raises
    ------
    ValueError
      If a value is not one of the types mentioned above.

    Examples
    --------

    >>> params = {'target': 'target',
                  'max_depth': [3,5,7],
                  'step_size': np.expon(.1)}
    >>> d = _random_choice(params)
    >>> d
    {'target': 'target', 'max_depth': 5, 'step_size': 0.5697}

    """
    model_params = {}
    for k, v in model_parameters.items():
        if type(v) is list:
            model_params[k] = random.choice(v)
        elif 'scipy' in str(type(v)):
            try:
                from scipy.stats import distributions
            except ValueError:
                raise ValueError('random_search requires scipy.stats')

            model_params[k] = v.rvs(1)[0]
        elif isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
            model_params[k] = v
        else:
            raise ValueError('Unrecognized search value for parameter %s' % k)
    return model_params


@_add_docstring(param_data=_DATA_DOCSTRING,
               param_model_factory=_MODEL_FACTORY_DOCSTRING,
               param_model_params=_MODEL_PARAMETERS_DOCSTRING,
               param_evaluator=_EVALUATOR_DOCSTRING,
               param_environment=_ENVIRONMENT_DOCSTRING,
               param_return_model=_RETURN_MODEL_DOCSTRING,
               param_perform_trial_run=_PERFORM_TRIAL_RUN_DOCSTRING,
               param_max_models=_MAX_MODELS_DOCSTRING,
               param_returns=_RETURNS_DOCSTRING)
def create(datasets,
           model_factory,
           model_parameters,
           evaluator=_default_evaluator,
           environment=None,
           return_model=True,
           perform_trial_run=True,
           max_models=10):
    """
    Evaluate model performance, in parallel, over a set of parameters, where
    the parameters are chosen randomly.

    Parameters
    ----------
    {param_data}
    {param_model_factory}
    {param_model_params}
        A user can also specify a random variable as the value for an argument.
        For each model, the parameter value will be sampled from this distribution.
        For a given scipy.distribution, v, each model will first call v.rvs(1)
        to sample a single value from the distribution.
        For example, 'step_size': scipy.stats.distribution.expon(.1)
        would choose step_size to be the result of calling the `rvs` method
        on the exponential distribution.

    {param_evaluator}
    {param_environment}
    {param_return_model}
    {param_perform_trial_run}
    {param_max_models}
    {param_returns}

    See Also
    --------
    graphlab.toolkits.model_parameter_search.create, graphlab.toolkits.model_parameter_search.manual_search.create

    Examples
    --------
    Perform a random search on a single train/test split.

    .. sourcecode:: python

        >>> import scipy.stats
        >>> sf = gl.SFrame()
        >>> sf['x'] = range(100)
        >>> sf['y'] = [0, 1]* 50
        >>> train, valid = sf.random_split(.5)
        >>> params = dict([('target', 'y'),
                           ('step_size', scipy.stats.distributions.expon(.1)),
                           ('max_depth', [5, 7])])
        >>> job = gl.random_search.create((train, valid),
                                        gl.boosted_trees_regression.create,
                                        params)
        >>> job.get_results()

    Perform a random search on a k-fold split.

    .. sourcecode:: python

        >>> folds = gl.cross_validation.KFold(sf, 5)
        >>> params = dict([('target', 'y'),
                           ('step_size', scipy.stats.distributions.expon(.1)),
                           ('max_depth', [5, 7])])
        >>> job = gl.random_search.create(folds,
                                          gl.boosted_trees_classifier.create,
                                          params)
        >>> job.get_results()

    """

    # Construct an iterable of all the desired free_param settings.
    model_param_list = []
    for _ in range(max_models):
        model_params = _random_choice(model_parameters)
        model_param_list.append(model_params)

    return _create_model_search(datasets,
                                model_factory,
                                model_param_list,
                                strategy='random',
                                evaluator=evaluator,
                                environment=environment,
                                return_model=return_model,
                                perform_trial_run=perform_trial_run)

