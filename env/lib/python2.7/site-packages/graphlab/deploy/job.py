from re import compile as _compile
from re import sub as _sub

import graphlab as _gl
from . import environment as _environment
from . import _executionenvironment as _env
from . import _job
from . import _task
from datetime import datetime as _datetime
from graphlab.util import _raise_error_if_not_function, _raise_error_if_not_of_type
from graphlab.connect import _get_metric_tracker

import logging as _logging
import uuid as _uuid

__LOGGER__ = _logging.getLogger(__name__)
_job_name_checker = _compile('^[a-zA-Z0-9-_]+$')

def _validate_job_create_args(function, name, environment):
    """ Validate the arguments for job.create and map_job.create
    """
    __LOGGER__.info("Validating job.")
    _raise_error_if_not_of_type(environment,
               [type(None), str, _environment._Environment],  'environment')
    _raise_error_if_not_of_type(name,
               [type(None), str], 'name')

    if name is not None and not _job_name_checker.match(name):
        raise ValueError('Job name can only contain digits, characters, "-" and "_".')

    # Setup the env
    if not environment:
        try:
            environment = _gl.deploy.environments['async']
        except KeyError:
            __LOGGER__.info("Creating a LocalAsync environment called 'async'.")
            try:
                environment = _environment.LocalAsync('async')
            except KeyError:
                environment = _gl.deploy.environments['async']
    else:
        if isinstance(environment, str):
            __LOGGER__.debug("Loading environment: %s" % environment)
            environment = _gl.deploy.environments[environment]

    # Clone to prevent the user's environment to reflect changes.
    return function, name, environment


def create(function, name=None, environment=None, **kwargs):
    """
    Execute arbitrary functions in a remote environment.

    The job is specified as a function. All functions that are called from
    within the function are automatically captured. By default, this method will
    kick off asynchronous work, and return a Job object to monitor/manage that
    work.

    Parameters
    ----------
    function : function
        Function to be executed in this Job, with arguments to pass to this
        function specified by `kwargs`.

    name : str, optional
        Name for this execution (names the returned Job). If set to None, then
        the name of the job is set to the name of the function with a time-stamp.
        Valid characters in job name include: digits, characters, '-' and '_'.

    environment : :class:`~graphlab.deploy.hadoop_cluster.HadoopCluster` | :class:`~graphlab.deploy.ec2_cluster.Ec2Cluster` | :class:`~graphlab.deploy.LocalAsync`, optional
        Optional environment for execution. If set to None, then a `LocalAsync`
        by the name `async` is created and used. This will execute the code in
        the background on your local machine.

    kwargs:
        Function kwargs that are passed to the function for execution.

    Returns
    -------
    job : :py:class:`~graphlab.deploy.Job`
        Used for monitoring and managing the execution of the Job.

    See Also
    --------
    graphlab.deploy.map_job.create, graphlab.deploy.Job

    Examples
    --------
    Let us start out with a simple example to execute a function that can
    add two numbers.

    .. sourcecode:: python

        # Define a function
        def add(x, y):
            return x + y

        # Create a job.
        job = graphlab.deploy.job.create(add, x=1, y=1)

        # Get results from the execution when ready. This call waits for the
        # job to complete before retrieving the results.
        >>> print job.get_results()
        2

    Exceptions within the function calls can be captured as follows:

    .. sourcecode:: python

        def add(x, y):
            if x and y:
                return x + y
            else:
                raise ValueError('x or y cannot be None')

        # Job execution capture the exception raised by the function.
        job = graphlab.deploy.job.create(add, x=1, y=None)

        # Get results from the execution when ready. This call waits for the
        # job to complete before retrieving the results.
        >>> print job.get_results()
        None

        # Get the exceptions raised from this execution by calling
        # job.get_metrics()
        >>> print job.get_metrics()
        +-----------+--------+------------+----------+-----------------------+
        | task_name | status | start_time | run_time |   exception_message   |
        +-----------+--------+------------+----------+-----------------------+
        |    add    | Failed | 1427928898 |   None   | x or y cannot be None |
        +-----------+--------+------------+----------+-----------------------+
        +-------------------------------+
        |      exception_traceback      |
        +-------------------------------+
        | Traceback (most recent cal... |
        +-------------------------------+
        [1 rows x 6 columns]


    If a function requires a package to be installed, the function can be
    annotated with a decorator.

    .. sourcecode:: python

        def my_function(number = 10):
            import names
            people = [names.get_full_name() for i in range(number)]
            sf = graphlab.SFrame({'names':people})
            return sf

        job = graphlab.deploy.job.create(my_function)

        >>> print job.get_results()

        Columns:
                names    str

        Data:
        +-------------------+
        |       names       |
        +-------------------+
        |   Annette Logan   |
        |   Nancy Anthony   |
        |  Tiffany Zupancic |
        |    Andre Coppin   |
        |     Robert Coe    |
        |    Donald Dean    |
        |    Lynne Bunton   |
        |   John Sartwell   |
        |   Peter Nicholas  |
        | Chester Rodriguez |
        +-------------------+
        [10 rows x 1 columns]

    Complex functions that require SFrames, GraphLab models etc. can be deployed
    with ease. All additional state required by the function are automatically
    captured.

    .. sourcecode:: python

        GLOBAL_CONSTANT = 10

        def foo(x):
            return x + 1

        def bar(x):
            return x + 2

        def my_function(x, y):
            foo_x = foo(x)
            bar_y = bar(y)
            return foo_x + bar_y + GLOBAL_CONSTANT

        # Automatically captures all state needed by the deployed function.
        job = graphlab.deploy.job.create(my_function, x = 1, y = 1)

        >>> print job.get_results()
        15

    You can execute the same job remotely by passing a different environment.

    .. sourcecode:: python

        # Define a function
        def add(x, y):
            return x + y

        # Define an EC2 environment
        ec2 = graphlab.deploy.Ec2Config()

        # Create an EC2 cluster object
        c = graphlab.deploy.ec2_cluster.create('my_cluster', 's3://bucket/path', ec2)

        # Create a job.
        job = graphlab.deploy.job.create(add, environment=c, x=1, y=1)

        >>> print job.get_results()
        2

    Notes
    -----
    - When an exception is raised within the deployed function,
      :func:`~graphlab.deploy.Job.get_results` returns None.

    - For asynchronous jobs, :func:`~graphlab.deploy.Job.get_results` is a
      blocking call which will wait for the job execution to complete
      before returning the results.

    """
    _session = _gl.deploy._default_session


    _raise_error_if_not_function(function)

    _get_metric_tracker().track('jobs.job')

    # Name the job
    now = _datetime.now().strftime('%b-%d-%Y-%H-%M-%S')
    function_name = _sub('[<>]','',function.__name__)

    name = '%s-%s' % (function_name, now) if not name else name
    # Validate args
    function, name, environment = _validate_job_create_args(function,
                                                            name, environment)
    while _session.exists(name, _job.Job._typename):
        rand = str(_uuid.uuid4())[:5]
        old_name = name
        name = "%s-%s" % (name, rand)
        __LOGGER__.info("A job with name '%s' already exists. "
                        "Renaming the job to '%s'." % (old_name, name))

    # Setup the task & job
    task = _task.Task(function,function_name)
    task.set_inputs(kwargs)
    job = _job.Job(name, stages=[[task]], environment=environment,
                                        final_stage=task)
    # Setup the env.
    __LOGGER__.info("Validation complete. Job: '%s' ready for execution." % name)
    exec_env = _env._get_execution_env(environment)
    job = exec_env.run_job(job)

    # Save the job and return to user
    if not isinstance(environment, _environment.Local):
        __LOGGER__.info("Job: '%s' scheduled." % name)
    else:
        __LOGGER__.info("Job: '%s' finished." % name)

    _session.register(job)
    _session.save(job)
    return job
