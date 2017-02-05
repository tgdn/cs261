from ._model_parameter_search import _create_model_search
from ._model_parameter_search_evaluators import default_evaluator as _default_evaluator
from itertools import product as _product
from ._model_parameter_search import _add_docstring, _DATA_DOCSTRING, _MODEL_FACTORY_DOCSTRING, _MODEL_PARAMETERS_DOCSTRING, _EVALUATOR_DOCSTRING, _ENVIRONMENT_DOCSTRING, _RETURN_MODEL_DOCSTRING, _MAX_MODELS_DOCSTRING, _RETURNS_DOCSTRING, _PERFORM_TRIAL_RUN_DOCSTRING


def _get_all_parameters_combinations(parameters):
    """
    Takes a dictionary where the keys are parameter names. The value of a key
    is a list of all possible values parameter.

    Returns a list of all possible parameter combinations. Each parameter set
    is a dictionary.

    For example an input of {'foo':[1,2], 'bar': {'x': ['a','b']}}
    will produce
    [{'foo':1, 'bar':{'x': 'a'}}, {'foo':1, 'bar':{'x': 'b'}},
     {'foo':2, 'bar':{'x': 'a'}}, {'foo':2, 'bar': {'x': 'b'}}]
    """

    # Get all possible combinations
    parameter_names = list(parameters.keys())
    arg_list = []
    for i in parameter_names:
        val = parameters[i]
        if isinstance(val, dict):
            arg_list.append(_get_all_parameters_combinations(val))
        elif not isinstance(val, list):
            arg_list.append([val])
        else:
            arg_list.append(val)
    param_iter = _product(*arg_list)

    # Construct the output
    result = []
    for param_tuple in param_iter:
        param_dict = {}
        for i in range(len(param_tuple)):
            cur_arg_name = parameter_names[i]
            cur_arg_value = param_tuple[i]
            param_dict[cur_arg_name] = cur_arg_value
        result.append(param_dict)

    return result


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
           perform_trial_run=True):
    """
    Evaluate model performance, in parallel, over a grid of parameters.

    Parameters
    ----------
    {param_data}
    {param_model_factory}
    {param_model_params}
        The collection of all combinations of valid parameter values defines a
        grid of model parameters that will be considered.

    {param_evaluator}
    {param_environment}
    {param_return_model}
    {param_perform_trial_run}
    {param_returns}

    See Also
    --------
    graphlab.toolkits.model_parameter_search.create,
    graphlab.toolkits.model_parameter_search.random_search.create,
    graphlab.toolkits.cross_validation.cross_val_score

    Examples
    --------

    Perform a grid search on a single train/test split.

    >>> train, valid = sf.random_split()
    >>> params = dict([('target', 'Y'),
                       ('step_size', [0.01, 0.1]),
                       ('max_depth', [5, 7])])
    >>> job = gl.grid_search.create((train, valid),
                                    gl.boosted_trees_classifier.create,
                                    params)
    >>> job.get_results()

    Perform a grid search on a k-fold split.

    >>> folds = gl.cross_validation.KFold(sf, 5)
    >>> params = dict([('target', 'Y'),
                       ('step_size', [0.01, 0.1]),
                       ('max_depth', [5, 7])])
    >>> job = gl.grid_search.create(folds,
                                    gl.boosted_trees_classifier.create,
                                    params)
    >>> job.get_results()
    """

    search_space = _get_all_parameters_combinations(model_parameters)

    return _create_model_search(datasets,
                                model_factory,
                                search_space,
                                strategy='grid',
                                evaluator=evaluator,
                                environment=environment,
                                return_model=return_model,
                                perform_trial_run=perform_trial_run)

