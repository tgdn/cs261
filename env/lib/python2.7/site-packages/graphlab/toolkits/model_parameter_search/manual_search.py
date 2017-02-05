from ._model_parameter_search_evaluators import default_evaluator as _default_evaluator
from ._model_parameter_search import _create_model_search
from ._model_parameter_search import _add_docstring, _DATA_DOCSTRING, _MODEL_FACTORY_DOCSTRING, _MODEL_PARAMETERS_DOCSTRING, _EVALUATOR_DOCSTRING, _ENVIRONMENT_DOCSTRING, _RETURN_MODEL_DOCSTRING, _RETURNS_DOCSTRING, _PERFORM_TRIAL_RUN_DOCSTRING

@_add_docstring(param_data=_DATA_DOCSTRING,
               param_model_factory=_MODEL_FACTORY_DOCSTRING,
               param_model_params=_MODEL_PARAMETERS_DOCSTRING,
               param_evaluator=_EVALUATOR_DOCSTRING,
               param_environment=_ENVIRONMENT_DOCSTRING,
               param_return_model=_RETURN_MODEL_DOCSTRING,
               param_perform_trial_run=_PERFORM_TRIAL_RUN_DOCSTRING,
               param_returns=_RETURNS_DOCSTRING)
def create(datasets,
           model_factory,
           model_parameters,
           evaluator=_default_evaluator,
           environment=None,
           return_model=True,
           perform_trial_run=True):
    """
    Evaluate model performance, in parallel, over a list of parameter
    combinations.

    Parameters
    ----------
    {param_data}
    {param_model_factory}

    model_parameters : list
      A list of dicts containing valid model parameter settings.

    {param_evaluator}
    {param_environment}
    {param_return_model}
    {param_perform_trial_run}
    {param_returns}

    See Also
    --------
    graphlab.toolkits.model_parameter_search.create,
    graphlab.toolkits.model_parameter_search.random_search.create

    Examples
    --------

    Fit models over a list of valid parameter settings.

    .. sourcecode:: python

        >>> import graphlab as gl
        >>> sf = gl.SFrame()
        >>> sf['x'] = range(100)
        >>> sf['y'] = [0, 1]* 50
        >>> factory = gl.boosted_trees_classifier.create
        >>> params = [dict([('target', 'y'), ('max_depth', 3)]),
                      dict([('target', 'y'), ('max_depth', 6)])]
        >>> job = gl.manual_search.create((training, validation),
                                          factory, params)

    """

    return _create_model_search(datasets,
                                model_factory,
                              model_parameters,
                              strategy='manual',
                              evaluator=evaluator,
                              environment=environment,
                              return_model=return_model,
                              perform_trial_run=perform_trial_run)


