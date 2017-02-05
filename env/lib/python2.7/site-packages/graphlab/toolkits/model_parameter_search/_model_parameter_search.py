import six as _six
import string
import sys as _sys
import time as _time
from array import array as _array
from datetime import datetime as _datetime
from collections import OrderedDict as _OrderedDict
from graphlab import aggregate as _agg
from graphlab import SFrame as _SFrame
from graphlab import SArray as _SArray
from graphlab.deploy import map_job as _map_job
from graphlab.connect import _get_metric_tracker
from ._model_parameter_search_evaluators import default_evaluator as _default_evaluator
import inspect as _inspect
import logging
import math

if _sys.version_info.major == 3:
    xrange = range


_DATA_DOCSTRING = """
    data : KFold | str | tuple | iterable of tuples
        The data to use when training and evaluating models.
        Typically this will be a (train, test) tuple, or the output from
        a method such as :py:class:`graphlab.toolkits.cross_validation.KFold` that
        creates an iterable over (train, test) pairs.

        When `model_factory` is a Graphlab Create model, the dataset
        may be an SFrame or a path to a saved SFrame.  When passing
        the path to the SFrame, the path can be one of the following:

        - **Locally**: Relative or absolute path.
        - **EC2**: An S3 path.
        - **Hadoop**: A full HDFS path including hostname.

        When the `model_factory` is a sklearn model, the dataset for
        the `data` argument should be a tuple of numpy arrays with the first
        element being the training features and second element being
        the target, e.g. (X, y), or None if no target is needed.
        """

_MODEL_FACTORY_DOCSTRING = """
    model_factory : function
        This is the function used to create the model. For example, to perform
        `model_parameter_search` using the GraphLab Create model
        :py:class:`graphlab.linear_regression.LinearRegression`, the
        `model_factory` is :py:func:`graphlab.linear_regression.create`.
        Supported GLC models include:

          * kmeans.create
          * logistic_classifier.create
          * boosted_trees_classifier.create
          * neuralnet_classifier.create
          * svm_classifier.create
          * linear_regression.create
          * boosted_trees_regression.create
          * ranking_factorization_recommender.create
          * factorization_recommender.create

        This argument can also be a scikit-learn model name, e.g.
        GradientBoostingClassifier as shown in the examples. The job
        will construct this model and call the fit method using the
        training_set. The parameters provided to the search method will be
        used when constructing the sklearn model. Supported sklearn
        models include:

          * SVC
          * LogisticRegression
          * GradientBoostingClassifier
          * GradientBoostingRegressor
          * RandomForestClassifier
          * RandomForestRegressor
          * ElasticNet
          * LinearRegression

        If a user-created function is provided, it must have a signature of
        `f(training_set, **kwargs)` and return a function. This function must
        work with the provided evaluation function. The Examples section
        contains an example of providing a custom function combined with
        a custom evaluator.
    """

_MODEL_PARAMETERS_DOCSTRING = """
    model_parameters : dict
        The params argument takes a dictionary containing parameters that will
        be passed to the provided model factory. Each (key, value) corresponds
        to an argument for the method and its possible values, respectively.
        If the value is an list, every value in that list must be valid
        input for the corresponding argument. Any values that are str, int,
        or floats are treated as a list containing a single value; specifying
        a dictionary with key "target" and value "y" means that "y" will be
        the chosen target every time the model is fit.

        On the other hand, for list-typed arguments (such as 'features') the
        provided value must be an iterable over valid argument values. For
        example, using 'features': [['col_a'], ['col_a', 'col_b']] would
        search over the two features sets; to use a single setting for this
        argument you need to use: 'features': [['col_a']].

        Some models have a parameter called validation_set that is used to
        monitor predictive performance on a validation set as the model is
        being trained. To skip this, you can add 'validation_set': None.
    """

_EVALUATOR_DOCSTRING = """
    evaluator : Function (model, training_set, validation_set) -> dict, optional
        The evaluation function takes as input the model, training and
        validation SFrames, and returns a dictionary of evaluation metrics
        where each value is a simple type, e.g. float, str, or int.

        .. sourcecode:: python

            # A sample evaluation function for KMeans.
            def custom_evaluator(model, train, test):
                mean_squared_distance = model['cluster_info']['sum_squared_distance'].mean()
                return {'mean_distance': mean_squared_distance}

        The default evaluator supports Recommender, ClassifierModel,
        RegressionModel, and TopicModel using the default evaluation metrics on
        training and validation set. For any sklearn models, we will compute
        the model's score method on the training and validation sets.
    """

_ENVIRONMENT_DOCSTRING = """
    environment : ::class:`~graphlab.deploy.hadoop_cluster.HadoopCluster` | :class:`~graphlab.deploy.ec2_cluster.Ec2Cluster` | :class:`~graphlab.deploy.LocalAsync`, optional
        Optional environment for execution. If set to None, then a `LocalAsync`
        by the name `async` is created and used. This will execute the code in
        the background on your local machine.
    """

_RETURN_MODEL_DOCSTRING = """
    return_model : bool, optional
        When `True`, collect all trained models created during the model parameter
        search. The list of models can be obtained via get_models method.
        If False, models will be discarded, and the
        ``models`` field is set to None. Set to `False` when the job is running
        remotely and model serialization is expensive.
    """

_PERFORM_TRIAL_RUN_DOCSTRING = """
    perform_trial_run : bool, optional
        If true, a job will be run using the first proposed parameter set in
        the search. If this job succeeds, the search continues as intended;
        otherwise, the failed job will be returned with a status message.
    """

_MAX_MODELS_DOCSTRING = """
    max_models : int, optional
        The maximum number of unique models to consider in the model parameter
        search.
    """

_RETURNS_DOCSTRING = """
    Returns
    -------
    out : :class:`.ModelSearchJob` object
        The job for the parameter search. This object can be used to track the
        progress of the parameter search.
    """


class _Formatter(string.Formatter):
    """
    Format {strings} that are within {brackets} as described in the doctring.
    """
    def get_value(self, key, args, kwargs):
        if hasattr(key, "__mod__") and key in args:
            return args[key]
        elif key in kwargs:
            return kwargs[key]
        return '{%s}' % key


def _add_docstring(**format_dict):
    """
    __example_start = '''
       Examples
       ---------
    '''
    __create = '''
       >>> import graphlab as gl

       >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
       >>> data = graphlab.SFrame.read_csv(url)
       >>> data['target'] = (data['target'] == 'e')

       >>> train, test = data.random_split(0.8)
       >>> model = graphlab.boosted_trees.create(train, target='label', *args, **kwargs)
    '''

    @add_docstring(create = __create, example_start = __example_start)
    def predict(x, **kwargs):
      '''
      {example_start}{create}
      '''
      return x

    """
    def add_doc_string_context(func):
        wrapper = func
        formatter = _Formatter()

        if _sys.version_info.major == 2:
            docs = func.func_doc
        else:
            docs = func.__doc__

        wrapper.func_doc = formatter.format(docs, **format_dict)
        return wrapper
    return add_doc_string_context


def _combiner(**tasks):
    """
    Take the return values from each task, and return
    the combined result.

    The combined result is a tuple, where the first
    element is a list of models, and the second
    sframe is a summary sframe containing
    the searched parameters and the evaluation result.
    """
    # Concatenate output from all the tasks.
    models = []
    evaluations = []
    parameters = []
    metadatas = []
    for t in tasks.values():
        if t is not None:  # If an exception occurred, t is None
            models.append(t['model'])
            evaluations.append(t['evaluation'])
            parameters.append(t['parameters'])
            metadatas.append(t['metadata'])

    if all(m is None for m in models):
        models = None

    # SFrame contains all the evaluation results, one row per model
    if all(type(x) in (int, float, str, list, type(None))
           for x in evaluations):
        evaluation_sframe = _SFrame({'metric': evaluations})
    else:
        evaluation_sframe = _SArray(evaluations).unpack(
            column_name_prefix=None)

    # SFrame contains all metadata, one row per model
    if all(type(x) in (int, float, str, list, type(None))
           for x in metadatas):
        metadata_sframe = _SFrame({'metadata': metadatas})
    else:
        metadata_sframe = _SArray(metadatas).unpack(
            column_name_prefix=None)

    # SFrame contains all the tuning parameters, one row per model
    if all(x is None or len(x) == 0 for x in parameters):
        parameter_sframe = _SFrame(
            {'parameters': [None] * len(parameters)})
    else:
        parameter_sframe = _SArray(parameters).unpack(
            column_name_prefix=None)

    # Make a summary sframe concatenating horizontally the evalution_sframe
    # and paramter_sframe
    summary_sframe = _SFrame()
    param_columns = sorted(parameter_sframe.column_names())
    metric_columns = sorted(evaluation_sframe.column_names())
    metadata_columns = sorted(metadata_sframe.column_names())
    summary_sframe[param_columns] = parameter_sframe[param_columns]
    summary_sframe[metric_columns] = evaluation_sframe[metric_columns]
    summary_sframe[metadata_columns] = metadata_sframe[metadata_columns]
    return _OrderedDict([('models', models), ('summary', summary_sframe)])


def _raise_if_evaluator_return_is_not_packable(eval_result):
    if type(eval_result) in (int, float, str, list, _array):
        return
    try:
        _SArray([eval_result]).unpack(column_name_prefix=None)
    except:
        raise ValueError('Return of the evaluator must be a dict '
                         'with simple types.')


def _train_test_model(model_factory,
                      folds,
                      model_parameters,
                      evaluator,
                      return_model,
                      metadata):
    """
    This is the actual top level function that will be run (possibly remotely)
    to do the actual work of creating and evaluating models with
    different parameters.

    Parameters
    ----------
    model_factory : function
      same as model_factory from model_parameter_search

    folds : KFold

    model_parameters : dict
      dictionary of model parameters

    evaluator : function
      function that takes model, training_set, and test_set
      and return a dictionary of simple typed values as evaluation result

    return_model : bool
      If true, include the model object in the return value.

    metadata : dict
      Dictionary of metadata describing this task.

    Returns
    -------
    out: dict
      The return dictionary contains the following fields:

      - parameters : a dictionary of parameters being searched
      - model : the model object if `return_model` is True. None otherwise.
      - evaluation : the output of the evaluator function
      - metadata : the user-provided metadata for this run.
    """
    if 'fold_id' in metadata:
        fold_id = metadata['fold_id']
    else:
        fold_id = 0

    training_set, validation_set = folds[fold_id]

    if isinstance(training_set, str):
        training_set = _SFrame(training_set)

    if isinstance(validation_set, str):
        validation_set = _SFrame(validation_set)

    # Some parameters require a validation set is provided, e.g. early stopping.
    if _sys.version_info.major == 3:
        argspec = _inspect.getargspec(model_factory)
        args = argspec.args
    else:
        args, varargs, varkw, defaults = _inspect.getargspec(model_factory)

    if 'validation_set' in args and not 'validation_set' in model_parameters:
        model_parameters['validation_set'] = validation_set

    # Create the model
    model = model_factory(training_set, **model_parameters)

    # Remove validation_set from model_parameters before summarizing
    if 'validation_set' in model_parameters:
        del model_parameters['validation_set']

    # Pack results.
    result = {}
    result['parameters'] = model_parameters
    result['model'] = model if return_model else None

    # Evaluate results (capture exceptions).
    evaluate_result = evaluator(model, training_set, validation_set)

    # Return the results as dictionaries.
    if evaluate_result is not None:
        _raise_if_evaluator_return_is_not_packable(evaluate_result)
    result['evaluation'] = evaluate_result
    result['metadata'] = metadata
    return result


def _check_if_sklearn_factory(sklearn_module, params):
    """
    Create a factory method for a sklearn model. This first splits the
    provided parameters into those that apply to the model's constructor and
    those that apply to the fit method.

    This is used as an argument to _train_test_model by model_param_search.

    Parameters
    ----------
    sklearn_module : class
        This is a sklearn object that can be constructed and that has a `fit`
        method.

    params : dict
        A dictionary of model arguments that can be passed to the constructor.

    Returns
    -------
    out : function
        A function with signature f(training_set, **params) that constructs
        the appropriate model object and invokes the fit method with the
        provided training data.

    Raises
    ------
    ValueError
        If any of the keys in the provided params is not a valid argument of
        either the constructor or fit method for the provided sklearn object.
    """
    if 'sklearn' not in str(sklearn_module):
        return sklearn_module

    import inspect
    allowed_params = inspect.getargspec(sklearn_module.__init__).args

    def factory(training_set, **params):
        if not isinstance(training_set, tuple):
            raise ValueError(
                'The training_set for scikit-learn models should '
                'be a tuple containing features and target, i.e. (X, y). '
                'Instead training_set was a %s.' %
                type(training_set))

        for k, v in _six.iteritems(params):
            if k not in allowed_params:
                raise ValueError(
                    'Unexpected argument %s of value %s for '
                    'constructing/fitting a %s.' %
                    (k, v, sklearn_module.__name__))

        (X, y) = training_set
        m = sklearn_module(**params)
        m.fit(X, y)
        return m

    return factory


def _combine_mps_tasks(**tasks):
    # Concatenate output from all the completed tasks.
    models = []
    evaluations = []
    parameters = []
    metadatas = []
    status = {'Failed': 0, 'Completed': 0}
    for t in tasks.values():
        if t is not None:  # If an exception occurred, t is None
            models.append(t['model'])
            evaluations.append(t['evaluation'])
            parameters.append(t['parameters'])
            metadatas.append(t['metadata'])
            status['Completed'] += 1
        else:
            status['Failed'] += 1

    if all(m is None for m in models):
        models = None
    if all(x is None or len(x) == 0 for x in parameters):
        parameters = _SArray([None] * len(parameters), dtype=dict)
    evaluations = _SArray(evaluations, dtype=dict)
    parameters = _SArray(parameters, dtype=dict)
    metadatas = _SArray(metadatas, dtype=dict)

    summary = _SFrame({'metric': evaluations,
                       'metadata': metadatas,
                       'parameters': parameters})

    return _OrderedDict([('models', models),
                         ('summary', summary),
                         ('status', status)])

def _progress_single_combiner(results):
    res = results.unpack('metadata').unpack('parameters')
    metadatas = [c for c in res.column_names() if c.startswith('metadata')]
    context = [c for c in res.column_names() if c.startswith('parameters')]

    # Unpack metrics if possible
    try:
        res = res.unpack('metric')
        metrics = [c for c in res.column_names() if c.startswith('metric')]
    except:
        metrics = ['metric']

    metadatas.sort()
    context.sort()
    metrics.sort()
    res = res[metadatas + context + metrics]

    # Clean up column names
    for s in ['parameters.', 'metric.', 'metadata.']:
        res = res.rename({c: c.replace(s, '') for c in res.column_names()})

    return res


def _progress_multi_combiner(results):
    res = results.unpack('metadata').unpack('parameters')
    metadatas = [c for c in res.column_names() if c.startswith('metadata')]
    context = [c for c in res.column_names() if c.startswith('parameters')]

    # Unpack metrics if possible
    try:
        res = res.unpack('metric')
        metrics = [c for c in res.column_names() if c.startswith('metric')]
    except:
        metrics = ['metric']
        pass # Do nothing

    metadatas.sort()
    context.sort()
    metrics.sort()
    res = res[metadatas + context + metrics]

    # Get aggregators for all metrics
    aggs = {}
    for m in metrics:
        aggs['mean_' + m] = _agg.MEAN(m)
    for m in metadatas:
        aggs[m] = _agg.CONCAT(m)
    aggs['num_folds'] = _agg.COUNT()

    res = res.groupby(context, aggs)

    # Clean up column names
    for s in ['parameters.', 'metric.', 'metadata.']:
        res = res.rename({c: c.replace(s, '') for c in res.column_names()})

    return res


class ModelSearchJob(object):
    """
    The return object from a model parameter search.

    This model should not be constructed directly, instead use
    one of the methods for performing a model parameter search.
    :py:func:`model_parameter_search.create`,
    :py:func:`random_search.create`,
    :py:func:`manual_search.create`, and
    :py:func:`grid_search.create`.
    """

    def __init__(self, factory,
                 parameter_sets,
                 name,
                 strategy=None,
                 environment=None,
                 return_model=True):
        """
        Constructor for a ModelSearchJob.
        """

        self.factory = factory
        self.parameter_sets = parameter_sets
        self.name = name
        self.strategy = strategy
        self.return_model = return_model
        self.environment = environment

        def get_max_model_id(parameter_sets):
            max_model_id = 0
            for ps in parameter_sets:
                model_id = ps['metadata']['model_id']
                max_model_id = max(model_id, max_model_id)
            return max_model_id

        self.max_model_id = get_max_model_id(parameter_sets)

        # Create batches of parameter sets
        def chunks(l, n):
            """
            Yield successive n-sized chunks from l.
            """
            for i in xrange(0, len(l), n):
                yield l[i:i+n]

        # Tuning parameter for dividing jobs into batches
        batch_size = max(10, int(math.ceil(len(parameter_sets) / 3.0)))
        parameter_batches = [c for c in chunks(parameter_sets, batch_size)]

        # Construct jobs
        self.jobs = []
        for i, parameter_set in enumerate(parameter_batches):
            job_name = name + '%05d' % i
            job = _map_job.create(factory, parameter_set,
                                  name=job_name,
                                  environment=environment,
                                  combiner_function=_combine_mps_tasks)
            self.jobs.append(job)

    def cancel(self):
        """
        Cancels all jobs.
        """
        for j in self.jobs:
            j.cancel()

    def get_status(self):
        """
        Get the status of all jobs launched for this model search.
        """
        status = {'Completed': 0,
                  'Running'  : 0,
                  'Failed'   : 0,
                  'Pending'  : 0,
                  'Canceled' : 0}
        for j in self.jobs:
            job_status = j.get_status(_silent=True)

            if job_status == 'Completed':
                result = j.get_results()

                # Increment overall status with the map_job's status
                for k, v in _six.iteritems(result['status']):
                    status[k] += v
            else:
                # Otherwise assume all tasks have the same status as the job
                status[job_status] += len(j._stages[0])

        return status

    def get_metrics(self):
        """
        Retrieves the metrics for all of the jobs.
        """
        if len(self.jobs) == 0:
            raise ValueError("No jobs have been created for this search.")

        # Get all results for jobs.
        metrics = []
        for j in self.jobs:
            if j is not None:
                status = j.get_status()
                metric = j.get_metrics()
                if status == 'Completed' and metrics is not None:
                    metric['job_name'] = j.name
                    metrics.append(metric)

        if len(metrics) == 0:
            raise ValueError("No jobs have created metrics for this search. "
                             "This may be because they are still pending.")

        # Concatenate results
        sf = metrics[0]
        for s in metrics[1:]:
            sf.append(s)
        return sf

    def get_models(self, wait=True):
        """
        Get a list of the models that have been fit during the search.

        Parameters
        ----------
        wait : bool
          If True, does not return a value until all jobs are completed.

        Returns
        -------
        out : list

        Raises
        ------
        ValueError
          Raised if this method is called when the job was created with the
          return_models argument set to False.

        Examples
        --------

        """
        if not self.return_model:
            raise ValueError("This method is unavailable for this object "
                             "since the job was created using return_models "
                             "set to False. If you want model objects returned"
                             ", please create the job with this argument set "
                             "to True, and then call job.get_models().")

        # Get the model_id for each result and return a dictionary with
        # model_id keys and models as values.
        def get_model_by_id(result):
            ids = []
            for row in result['summary']:
                ids.append(row['metadata']['model_id'])
            models = {}
            ms = result['models']
            if ms is not None:
                for (i, m) in zip(ids, ms):
                    models[i] = m
            return models

        if wait:
            self._wait()

        combined_models = [None for i in range(self.max_model_id + 1)]
        for j in self.jobs:
            if j.get_status(_silent=True) == 'Completed':
                result = j.get_results()
                job_models = get_model_by_id(result)
                for i, m in _six.iteritems(job_models):
                    combined_models[i] = m

        if all([x is None for x in combined_models]):
            combined_models = None

        return combined_models

    def _wait(self):
        def is_done():
            """
            Count the number of jobs that are neither running nor pending
            """
            count = 0
            for k, v in _six.iteritems(self.get_status()):
                if k not in ['Running', 'Pending']:
                    count += v
            return count == len(self.parameter_sets)

        while not is_done():
            _time.sleep(2)

    def get_results(self, wait=True):
        """
        Get a current summary of completed jobs. The results are aggregated
        to be a summary for each unique set of parameters.

        Parameters
        ----------
        wait : bool
          If True, does not return a value until all jobs are completed.

        Returns
        -------
        out : SFrame

        """
        def _combine_sframes(summaries):
            summary = _SFrame()
            summary['metadata'] = _SArray(dtype=dict)
            summary['metric'] = _SArray(dtype=dict)
            summary['parameters'] = _SArray(dtype=dict)
            for s in summaries:
                summary = summary.append(s)
            return summary

        summaries = []

        if wait:
            self._wait()

        for j in self.jobs:
            if j.get_status(_silent=True) == 'Completed':
                result = j.get_results()
                summaries.append(result['summary'])

        summary = _combine_sframes(summaries)

        if len(summary) == 0:
            print('No valid results have been created from this search.')
            return None

        # If fold_id is a column, then a groupby is done on the results by
        # the _progress_multi_combiner method.
        has_multiple_folds = 'fold_id' in summary['metadata'][0].keys()
        if has_multiple_folds:
            summary = _progress_multi_combiner(summary)
        else:
            summary = _progress_single_combiner(summary)

        return summary

    def get_best_params(self, metric=None, ascending=False):
        """
        Return the parameters for the model with the best performance
        according to the provided metric. When no metric is provided, this
        method attempts to find a metric

        Parameters
        ----------
        metric : str
          The name of the column in the result summary that should be used to
          determine the best set of parameters.

        ascending : bool
          True if the lower values are better for the chosen metric.

        Returns
        -------
        out : dict
          Dictionary of parameter settings that were associated with the best
          observed value of the chosen metric.

        Examples
        --------

        >>> j = gl.model_parameter_search.create(data, gl.boosted_trees_classifier.create)
        >>> params = j.get_best_params()

        """

        res = self.get_models()

        if res is None:
            raise ValueError('No models detected. Please use `return_model` to'
                ' be True when creating a ModelSearchJob.')


        results = self.get_results()

        # If no metric is provided, then the best parameters will be chosen by
        # sorting the results using the first column found from the set:

        chosen_metrics = ['validation_rmse',
                          'validation_accuracy',
                          'validation_precision',
                          'validation_score',
                          'training_rmse',
                          'training_accuracy',
                          'training_precision',
                          'training_score',
                          'mean_validation_rmse',
                          'mean_validation_accuracy',
                          'mean_validation_precision',
                          'mean_validation_score',
                          'mean_training_rmse',
                          'mean_training_accuracy',
                          'mean_training_precision',
                          'mean_training_score']

        if metric is None:
            for m in chosen_metrics:
                if m in results.column_names():
                    metric = m
                    ascending = ('rmse' in metric)
                    break

        if metric is None:
            raise ValueError('Could not detect a valid column of model metrics.'
                'Please specify which column should be sorted.')

        results = results.sort(metric, ascending=ascending)

        # Get one of the models corresponding to the best parameter sets.
        best_model_id = results[0]['model_id']
        if isinstance(best_model_id, list):
            best_model_id = best_model_id[0]

        best_params = self.parameter_sets[best_model_id]['model_parameters']

        return best_params


    def __str__(self):
        status = self.get_status()

        header = "Model parameter search\n-------------------------"
        strategy  = 'Strategy            : %s' % self.strategy
        num_params= 'Num. combinations   : %s' % len(self.parameter_sets)

        stats = 'Current status\n-------------------------'
        if 'Completed' in status:
            stats += '\nCompleted           : %s' % status['Completed']
        if 'Running' in status:
            stats += '\nRunning             : %s' % status['Running']
        if 'Pending' in status:
            stats += '\nPending             : %s' % status['Pending']
        if 'Failed' in status:
            stats += '\nFailed              : %s' % status['Failed']
        if 'Canceled' in status:
            stats += '\nCanceled            : %s' % status['Canceled']

        jobs_out = 'Jobs\n-------------------------'
        for job in self.jobs:
            jobs_out += '\n' + job.name

        help_list = 'Help\n-------------------------'
        help_list += '\nGet status          : self.get_status()'
        help_list += '\nGet exceptions      : self.get_metrics()'
        help_list += '\nGet a single job    : self.jobs[i]'

        header_out = '\n'.join([header, strategy, num_params])
        out = header_out
        out += '\n\n%s' % stats
        out += '\n\n%s' % jobs_out
        out += '\n\n%s' % help_list

        return out


    def __repr__(self):
        return self.__str__()

    def summary(self):
        """
        Print a summary of the current progress of the jobs created for this
        search.
        """
        print(self.__repr__())



def _create_model_search(datasets,
                         model_factory,
                         model_parameters,
                         strategy=None,
                         evaluator=_default_evaluator,
                         environment=None,
                         return_model=True,
                         perform_trial_run=True):

    from graphlab.toolkits.cross_validation import KFold as _KFold

    track_message = 'jobs.model_parameter_search.{}'.format(strategy.replace(' ', '_'))
    if 'sklearn' in str(model_factory):
        track_message += '.sklearn'

    _get_metric_tracker().track('jobs.model_parameter_search.{}'.format(strategy))

    now = _datetime.now().strftime('%b-%d-%Y-%H-%M-%S')
    job_name = "Model-Parameter-Search-%s" % now

    if isinstance(datasets, _SFrame):
        folds = [(datasets, None)]
    elif isinstance(datasets, tuple):
        if len(datasets) != 2:
            raise ValueError("Provided dataset tuple must be train/test pair.")
        folds = [datasets]
    else:
        folds = datasets

    if (not isinstance(folds, _KFold)):
        folds = _KFold.from_list(folds)

    num_folds = folds.num_folds
    include_fold_id = num_folds > 1

    params = []
    model_id = 0
    for model_params in model_parameters:

        for fold_id in range(num_folds):
            metadata = {'model_id': model_id}
            if include_fold_id:
                metadata['fold_id'] = fold_id
            model_id += 1

            params.append({
                'model_factory': _check_if_sklearn_factory(model_factory, model_params),
                'model_parameters': model_params,
                'folds': folds,
                'evaluator': evaluator,
                'return_model': return_model,
                'metadata': metadata
            })

    # Run a trial job on the first set of parameters.
    if len(params) == 0:
        raise ValueError("No valid (parameter, data) pairs could be constructed.")

    # We perform a trial run, if trial run failed, we return the trial run
    # result for user to inpspect error.
    # If trial run succeeds, we preceed with real job submission.
    # Any exception happened to job submission will be raised
    # If an EC2 beta cluster is created, it will be scheduled to terminate
    # before raise or return.
    _logger = logging.getLogger(__name__)
    if perform_trial_run:
        trial_params = [params[0]]

        # Make sure to shut down the new cluster if any issues occur when
        # creating jobs.
        trial_run = ModelSearchJob(_train_test_model,
                                    trial_params,
                                    name=job_name,
                                    strategy=strategy,
                                    environment=environment,
                                    return_model=return_model)

        # Wait until the job exits pending state.
        trial_status = trial_run.get_status()
        if trial_status['Pending'] == 1:
            trial_run._wait()

        # If things failed, show a warning.
        trial_status = trial_run.get_status()
        if trial_status['Completed'] != 1 or sum(trial_status.values()) != 1:
            _logger.warning("Trial run failed prior to launching model parameter search.  "
                            "Please check for exceptions using get_metrics() on the "
                            "returned object.")
            return trial_run

    # We are done with trial_run continue with real job
    m = ModelSearchJob(_train_test_model,
                       parameter_sets=params,
                       strategy=strategy,
                       name=job_name,
                       environment=environment,
                       return_model=return_model)
    return m
