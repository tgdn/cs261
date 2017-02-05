from copy import copy as _copy
import uuid as _uuid

from datetime import datetime as _datetime
import logging as _logging

from re import sub as _sub
import graphlab as _gl
from graphlab.connect import _get_metric_tracker
from . import _task
from . import job as _job
from . import _job as __job
from . import _executionenvironment as _env
from . import environment as _environment

from graphlab.util import _raise_error_if_not_function

__LOGGER__ = _logging.getLogger(__name__)

def _generate_mapjob_tasks(task_prototype, param_set):
    '''
    Creates len(param_set) number of tasks. Each of which is a clone of
    task_prototype with a different element of param_set set as its parameters.
    The output of each task will be set to a temporary locations.

    Returns a list of tasks. Also returns
    the mapping of param values to output location.
    '''
    param_set = _copy(param_set)
    param_set.reverse() # Preserve order; we'll be 'popping' in reverse order.

    tasks = []

    num_in_cur_step = 0
    tasks.append([])
    while param_set:

        cur_params = param_set.pop()
        cur_name = '-'.join([task_prototype.name, str(0),
                                                        str(num_in_cur_step)])
        cur_task = task_prototype.clone(cur_name)
        cur_task.set_inputs(cur_params)
        tasks[-1].append(cur_task)
        num_in_cur_step += 1

    return tasks

def create(function, parameter_set, name=None, environment=None,
           combiner_function=None):
    '''
    Distributed execution of a function once for each entry in the parameter_set.

    Similar to the map() function in python, this method `maps` a single function
    to each provided set of parameters. The results are a list corresponding to each
    parameter in `parameter_set`.

    Parameters
    ----------
    function : function
        Function to be executed, with arguments to pass to this
        function specified by parameter_set.

    parameter_set : iterable of dict
        Each element of the list corresponds to an evaluation of the function
        with the dictionary argument.

    name : str, optional
        Name for the returned Job. If set to None, then the name of the Job is
        set to the name of the function with a timestamp.

    environment : :class:`~graphlab.deploy.hadoop_cluster.HadoopCluster` | :class:`~graphlab.deploy.ec2_cluster.Ec2Cluster` | :class:`~graphlab.deploy.LocalAsync`, optional
        Optional environment for execution. If set to None, then a `LocalAsync`
        by the name `async` is created and used. This will execute the code in
        the background on your local machine.

    combiner_function : function (kwargs -> object), optional
        An optional function that will be run once at end of the map_job. The
        combiner function will have access to all previous map job results. If
        a combiner is provided, only the output of the combiner will be reported.
        The input type of the combiner is `kwargs` where the values (in order)
        correspond to the values of the results from the map.

    Returns
    -------
    job : :py:class:`~graphlab.deploy.Job`
        The job for the map_job, which was run using the `environment`
        parameter. This object can be used to track the progress of
        map_job work.

    See Also
    --------
    graphlab.deploy.job.create, graphlab.deploy.Job

    Notes
    -----
    - The map job achieves the same behavior as `results = map(func, args)`

    Examples
    --------
    Let us start out with a simple example to execute a function that can
    add two numbers over 2 sets of arguments.

    .. sourcecode:: python

      # Define the function.
      def add(x, y):
          return x + y

      # Create a map-job
      params = [{'x': 1, 'y': 2}, {'x': 10, 'y': -1}]
      job = graphlab.deploy.map_job.create(add, params)

      # Get results from the execution when ready. This call waits for the
      # job to complete before retrieving the results.
      >>> print job.get_results()
      [3, 9]

    Exceptions within the function calls can be captured as follows:

    .. sourcecode:: python

        def add(x, y):
            if x and y:
                return x + y
            else:
                raise ValueError('x or y cannot be None')

        params = [{'x': 1, 'y': 2}, {'x': 10, 'y': None}]
        job = graphlab.deploy.map_job.create(add, params)

        # Get results from the execution when ready.
        >>> print job.get_results()
        [3, None]

        # Get the exceptions raised from this execution by calling
        # job.get_metrics()
        >>> print job.get_metrics()
        +-----------+-----------+------------+-------------------+-----------------------+
        | task_name |   status  | start_time |      run_time     |   exception_message   |
        +-----------+-----------+------------+-------------------+-----------------------+
        |  add-0-0  | Completed | 1427931034 | 3.81469726562e-05 |                       |
        |  add-1-0  |   Failed  | 1427931034 |        None       | x or y cannot be None |
        +-----------+-----------+------------+-------------------+-----------------------+
        +-------------------------------+
        |      exception_traceback      |
        +-------------------------------+
        |                               |
        | Traceback (most recent cal... |
        +-------------------------------+
        [2 rows x 6 columns]


    Use the combiner function to perform aggregations on the results.

    .. sourcecode:: python

      # Combiner to combine all results from the map.
      def max_combiner(**kwargs):
          return max(kwargs.values())

      # The function being mapped to the arguments.
      def add(x, y):
           return x + y

      # Create a map-job.
      params = [{'x': 1, 'y': 2}, {'x': 10, 'y': -1}]
      job = graphlab.deploy.map_job.create(add, params,
                                    combiner_function = max_combiner)

      # Get results. (Applies the combiner on the results of the map.)
      >>> print job.get_results()
      9
    '''

    _get_metric_tracker().track('jobs.map_job', properties={
        'num_tasks':len(parameter_set),
        'has_combiner':combiner_function is not None
        })

    _session = _gl.deploy._default_session

    job = _create_map_job(function, parameter_set, name, environment,
           combiner_function, _job_type = 'PIPELINE')

    # Setup the env.
    __LOGGER__.info("Validation complete. Job: '%s' ready for execution" % job.name)
    exec_env = _env._get_execution_env(environment)
    job = exec_env.run_job(job)

    # Save the job and return to user
    if not isinstance(environment, _environment.Local):
        __LOGGER__.info("Job: '%s' scheduled." % job.name)
    else:
        __LOGGER__.info("Job: '%s' finished." % job.name)

    _session.register(job)
    _session.save(job)
    return job

def _create_map_job(function, parameter_set, name=None, environment=None,
           combiner_function=None, _job_type = 'PIPELINE'):

    _raise_error_if_not_function(function)

    # Name the job
    now = _datetime.now().strftime('%b-%d-%Y-%H-%M-%S')
    function_name = _sub('[<>]','',function.__name__)

    name = '%s-%s' % (function_name, now) if not name else name

    # Validate args
    function, name, environment = _job._validate_job_create_args(function, name,
                                                                 environment)
    _session = _gl.deploy._default_session
    while _session.exists(name, __job.Job._typename):
        rand = str(_uuid.uuid4())[:5]
        old_name = name
        name = "%s-%s" % (name, rand)
        __LOGGER__.info("A job with name '%s' already exists. "
                        "Renaming the job to '%s'." % (old_name, name))

    # Convert SFrame to a dict
    if not parameter_set:
        raise RuntimeError('An empty parameter_set was given. Nothing to do.')

    # If parameter set is a generator/SFrame, make sure it gets expanded out.
    parameter_set_copy = []
    for i in parameter_set:
        if not isinstance(i, dict):
            raise TypeError("'parameter_set' has to be an iterable of dictionary."
                       " For void functions, use an empty dictionary as inputs.")
        parameter_set_copy.append(i)

    # Create the task.
    task_prototype = _task.Task(function,function_name)
    for_each_iterations  = _generate_mapjob_tasks(task_prototype, parameter_set_copy)

    # List of outputs for the final step.
    if not combiner_function:
        list_of_tasks = for_each_iterations[0]
    else:
        combiner = _task.Task(combiner_function)

        # The input to this task is all other tasks
        task_name_to_task = {}
        for stage in for_each_iterations:
            for t in stage:
                task_name_to_task[t.name] = t
        combiner.set_inputs_from_task(task_name_to_task)

        for_each_iterations.append([combiner])
        list_of_tasks = combiner

    # Create the job
    job = __job.Job(name, stages=for_each_iterations, environment=environment,
                              final_stage=list_of_tasks, _job_type = _job_type)
    return job
