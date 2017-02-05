import graphlab as _gl
import graphlab.canvas as _canvas
from . import _internal_utils
from ._task import Task as _Task
from ._artifact import Artifact as _Artifact
import time as _time
import shutil as _shutil

from graphlab.util import _raise_error_if_not_of_type, \
    _make_temp_directory
from graphlab import _gl_pickle as gl_pickle
from uuid import uuid4 as _uuid

from graphlab.util import file_util as _file_util
from .. import _minipsutil as _psutil
import requests as _requests
import os as _os
import datetime as _datetime
import json as _json
import tempfile as _tempfile
import logging as _logging


__LOGGER__ = _logging.getLogger(__name__)

class Job(_Artifact):
    """
    Monitor/Manage code executing on remote environments.

    Job objects should not be instantiated directly, and are intended to be
    created using :func:`graphlab.deploy.job.create` or
    :func:`graphlab.deploy.map_job.create`.

    See Also
    --------
    graphlab.deploy.job.create, graphlab.deploy.map_job.create

    """

    #  Version 1: GLC 1.3
    #  --------------------
    #  Everything starts from scratch here. Everything before V1.3 is not version
    #  aware and will not be backwards compatible.
    #
    #  Version 2: GLC 1.4
    #  --------------------
    #  We break backward compatibility again!! It is not version compatible with
    #  GLC 1.3
    _JOB_VERSION = 2

    _typename = 'Job'

    # Static definition
    _num_tasks = 0
    _stages = [[]]
    _final_stage = None
    _packages = set([])
    environment = None
    """ environment used to run this job """

    # Runtime settings
    _exec_dir = None
    _pid = None
    _task_output_paths = {}
    _status = 'Unknown'
    _task_status = None
    _metrics = []

    @classmethod
    def get_path_join_method(cls):
        return _os.path.join

    def __init__(self, name, stages=[[]], final_stage=None, environment=None,
                 _exec_dir=None, _task_output_paths=None, _job_type = 'PIPELINE'):
        """
        Construct a job.

        Parameters
        ----------
        name : str
            Name of this Job, must be unique.

        stages: list[list[Task]]
            Collection of task(s) to be executed.

        final_stage : list[task] | task
            Collection of task(s) whose outputs are to be returned._

        environment : Environment, optional
            Environment used for this execution. See
            :py:class:`~graphlab.deploy.environment.LocalAsync` for an example
            environment.

        """
        _raise_error_if_not_of_type(name, [str], 'name')
        _raise_error_if_not_of_type(stages, [list], 'stages')
        _raise_error_if_not_of_type(final_stage,
                                [list, _Task, type(None)], 'final_stage')

        self.name = name
        self.environment = environment

        self._stages = stages
        self._num_tasks = 0
        self._status = 'Pending'
        self._start_time = None
        self._end_time = None
        self._error = None

        self._job_type = _job_type

        # Set the packages
        self._packages = set()
        for task in self._stages:
            for t in task:
                self._num_tasks += 1
                self._packages.update(t.get_required_packages())


        self._final_stage = final_stage
        self._task_status = {}

        self._session = _gl.deploy._default_session
        if not _exec_dir:
            relative_path = "job-results-%s" % str(_uuid())
            self._exec_dir = self.get_path_join_method()(self._session.results_dir, relative_path)
        else:
            self._exec_dir = _exec_dir

        # Location where all the outputs for the tasks are saved.
        if not _task_output_paths:
            Job._update_exec_dir(self, self._exec_dir)
        else:
            self._task_output_paths = _task_output_paths

    @classmethod
    def _update_exec_dir(cls, job, exec_dir):
        '''
        Update job execution folder to the new one. As a result of this, the
        job's task output path are updated too.

        This should only be called before the job is started to run
        '''
        job._exec_dir = exec_dir

        # update task output path
        job._task_output_paths = {}
        for s, tasks in enumerate(job._stages):
            for i, t in enumerate(tasks):
                job._task_output_paths[t] = cls.get_path_join_method()(job._get_exec_dir(), 'output',
                  '%s-%d-%d-%s.gl' % (t.name, s, i, _time.time()))

    def _test_url(self,file_path):
        if _file_util.is_hdfs_path(file_path):
            return _file_util.hdfs_test_url(file_path,'e',self.environment.hadoop_conf_dir)
        if _file_util.is_s3_path(file_path):
            return _file_util.s3_test_url(file_path,self.environment.ec2_config.get_credentials())
        else:
            return _os.path.exists(file_path)

    def _get_commander_uri(self, silent = True):
        '''
        Read  commander information
        Returns
        -------
        str : None if the file does not exist, otherwise, the uri of the commander
        '''
        if not hasattr(self, '_commander_url') or not self._commander_url:
            port_path = '/'.join([self._exec_dir, 'commander_init.status'])
            self._commander_url = self._load_file_and_parse(port_path,
                                            self._parse_commander_port_file,
                                            silent)

        return self._commander_url

    def _remote_flush_logs(self):
        """
        Force flushing the logs in the commander and workers.
        """
        if not self._job_finished():
            get_log_path_request = "%s/flush_logs" % self._get_commander_uri()
            try:
                resp = _requests.post(get_log_path_request)
            except Exception as e:
                __LOGGER__.info("%s did not succeed: %s" % (get_log_path_request,e))
        else:
            __LOGGER__.info("job is already finished. flush_logs() is not going to be executed.")

    def _get_max_concurrent_tasks(self):
        return max(map(len, self._stages))

    def _get_version(self):
        return self._JOB_VERSION

    def _get_metadata(self):
        """
        Get the metadata that is managed by the session. This gets displayed
        in the Scoped session and is stored in the session's index file.

        """
        return {'Name': self.name,
                'Environment': self.environment.name,
                'Creation date': self._session_registration_date}

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : A GLUnpickler archive.
        """
        # Load the dump.
        obj = unpickler.load()

        # Construct a new object.
        new = cls(obj.name, obj._stages, obj._final_stage, obj.environment)
        cls._resolve_backward_compatibility(version, new._get_version())

        # Now copy over the useful parts of the dump.
        lst = ['_exec_dir', '_final_stage','_pid',
               '_task_output_paths', '_status', '_task_status',
               '_typename', 'environment', '_metrics', 'name',
               '_num_tasks', '_packages', '_stages',
               '_start_time', '_end_time']
        _internal_utils.copy_attributes(new, obj, lst)
        return new

    def _get_status(self, _silent = False):
        """Get job status from status file
        """
        status_file = self._exec_dir + '/status'
        status = self._load_file_and_parse(status_file,
                                           self._parse_status_file,
                                           _silent,test_url=False)
        return status if status is not None else 'Pending'

    def _get_metrics(self):
        """Get job runtime metrics from metrics file
        """
        metric_file = self._exec_dir + '/metrics'
        metrics = self._load_file_and_parse(metric_file,
                                            self._parse_metrics_file,
                                            silent=True,test_url=False)
        return metrics if metrics is not None else []

    def _parse_metrics_file(self, metric_file_name):
        """
        Read metrics file and parse the content, return the parsed metrics value
        """
        with open(metric_file_name, 'r') as f:
            metrics = _json.load(f)
        return metrics

    def _parse_commander_port_file(self, commander_port_file):
        '''
        Parse commander port file. Commander port file contains commander's
        DNS name and port
        Returns
        --------
        str : the Uri of the commander
        '''
        with open(commander_port_file,'r') as f:
            status_json = _json.load(f)
            port = status_json['port']
            host_name = status_json['host_name']

        return 'http://%s:%s' % (host_name, port)

    def _parse_status_file(self, status_file_name):
        """
        Read metrics file and parse the content, return the parsed metrics value
        """
        with open(status_file_name,'r') as f:
            status_json = _json.load(f)
            status = status_json['status']
            self._start_time = status_json.get('start_time', None)
            self._end_time = status_json.get('end_time', None)
            self._error = status_json.get('error', None)
        return status

    def _download_remote_folder_to_local(self, remote_path, silent=False):
        '''
        Download all files from remote path to local. Caller is responsible for
        cleaning up the local folder after finishing usage

        Returns the local temporary folder
        '''
        local_path = _tempfile.mkdtemp(prefix='job-results')

        try:
            if _file_util.is_hdfs_path(remote_path):

                _file_util.download_from_hdfs(
                    hdfs_path = remote_path,
                    local_path = local_path,
                    is_dir = True,
                    hadoop_conf_dir=self.environment.hadoop_conf_dir)

            elif _file_util.is_s3_path(remote_path):

                _file_util.download_from_s3(
                    s3_path = remote_path,
                    local_path = local_path,
                    is_dir = True,
                    aws_credentials = self.environment.ec2_config.get_credentials(),
                    silent = silent)
            else:
                raise RuntimeError("'%s' is not a supported remote path. Only S3 and HDFS"
                                    " remote path are supported" % remote_path)
        except:
            # Make sure we cleanup local files if we cannot successfully
            # download files
            if _os.path.isdir(local_path):
                _shutil.rmtree(local_path)

            raise

        return local_path

    def _load_file_and_parse(self, file_name, parser_func, silent=False, test_url=True):
        '''
        Read remote file to a local temporary file, and use parser_func
        to parse the content, returns the parsed result.

        This function is used for parsing state and progress files from
        either local, S3 or HDFS.

        If there is any exception happened, returns None
        '''
        file_is_local = _file_util.is_local_path(file_name)
        local_file_name = file_name if file_is_local else _tempfile.mktemp(prefix='job-status-')

        try:
            try:
                if test_url and not self._test_url(file_name):
                    if not silent:
                        __LOGGER__.info("File %s is not available yet." % file_name)
                    return None

                if _file_util.is_hdfs_path(file_name):

                    _file_util.download_from_hdfs(
                        hdfs_path = file_name,
                        local_path = local_file_name,
                        hadoop_conf_dir=self.environment.hadoop_conf_dir)

                elif _file_util.is_s3_path(file_name):

                    _file_util.download_from_s3(
                        s3_path = file_name,
                        local_path = local_file_name,
                        is_dir = False,
                        aws_credentials = self.environment.ec2_config.get_credentials(),
                        silent = silent)

            except Exception as e:
                # It is ok the status file is not ready yet as the job is getting prepared
                if not silent:
                    __LOGGER__.warning("Exception encountered when trying to download file from %s, error: %s" % (file_name, e))
                return None

            try:
                # parse the local file
                return parser_func(local_file_name)
            except Exception as e:
                __LOGGER__.info("Exception when parsing file %s. Error: %s" % (file_name, e))
                return None
        finally:
            if (not file_is_local) and _os.path.exists(local_file_name):
                _os.remove(local_file_name)

    def get_status(self, _silent = False):
        """
        Returns status information about execution.

        Returns
        -------
        status : str
            String representation of status.

        The status string can be one of the following

        'Pending'   : The job has not been scheduled yet.
        'Running'   : The job is currently running.
        'Completed' : The job is completed and the results are available.
        'Failed'    : The job execution failed.
        'Canceled'  : The job was cancelled by the user.
        'Unknown'   : An unknown exception occurred. See logs for more details.

        See Also
        ---------
        get_results

        Examples
        --------
        >>> print job.get_status()
        'Running'

        """
        if self._is_final_state(self._status):
            return self._status

        self._status = self._get_status(_silent = True)
        if self._is_final_state(self._status):
            self._finalize()
        return self._status

    def _deserialize_output(self, task):
        """
        Deserialize the output from a task.

        Parameters
        ----------
        Task definition of interest.

        Returns
        -------
        The output of the run-time task associated with the task definition.
        """
        filepath = self._task_output_paths[task]

        non_hdfs_file_path = filepath

        # Unpickler has no support for passing in additional HADOOP_CONF_DIR
        # so we download HDFS folder first before calling to unpickler
        if _file_util.is_hdfs_path(filepath):
            non_hdfs_file_path = _make_temp_directory("job_output_")
            _file_util.download_from_hdfs(filepath, non_hdfs_file_path,
                hadoop_conf_dir=self.environment.hadoop_conf_dir, is_dir = True)

        unpickler = gl_pickle.GLUnpickler(non_hdfs_file_path)

        # We cannot delete this temporary file path becaue SFrame lazily load
        # the content from disk. But the temporary folder will be removed
        # eventually when the python session goes away

        return unpickler.load()

    def get_map_results(self,_silent=True):
        """
        Retrieve the results for map job.

        This method is only applicable if the job is created through map_job.create
        API. In case of job success, get_map_results() returns the same value as
        get_results(). In case of job failure, get_map_results() can return partial
        results from individual map job. If a certain task failed, the result will
        be presented as None.

        See Also
        ---------
        get_status, get_results

        Examples
        ---------
        .. sourcecode:: python

            # Create a map job that partially fail
            def my_func(str_param):
                if not isinstance(str_param, str):
                    raise ValueException
                else:
                    return str_param

            import graphlab as gl
            parameter_set = [{'str_param':'hello'}, {'str_param':3}]
            j = gl.deploy.map_job.create(my_func, parameter_set=parameter_set)

            # This would raise exception
            j.get_results()

            # This returns ['str', None]
            j.get_map_results()
        """

        # cached results to avoid retrieving again
        if hasattr(self, '_map_results'):
            return self._map_results

        # Only works with map_job
        is_map_job = isinstance(self._final_stage, list)
        is_map_combiner_job = len(self._stages) == 2

        if not (is_map_job or is_map_combiner_job):
            raise TypeError('Only jobs created through map_job.create support get_map_results().')

        status = self.get_status(_silent=True)
        if status == 'Canceled':
            raise RuntimeError("The job execution was cancelled by the user. "\
                               "Cannot retrieve map results.")
        if status == 'Pending' or status == 'Unknown':
            if not _silent:
                __LOGGER__.info("The job execution is in %s state. Cannot retrieve map results" % status)
            return None

        if status == 'Running':
            return self._get_map_job_results(_silent)

        if self._is_final_state(status):
            self._map_results = self._get_map_job_results(_silent)
            return self._map_results

    def _get_map_job_results(self,_silent=True):
        '''
        Get results of all map jobs.

        Returns
        --------
        job outputs : list
          A list of results from the job. if a certain job failed, the result would be None
        '''

        result_folder = self.get_path_join_method()(self._exec_dir, 'output')
        __LOGGER__.info("Retrieving job results from %s..." % result_folder)
        if _file_util.is_local_path(result_folder):
            local_folder = result_folder
        else:
            local_folder = self._download_remote_folder_to_local(result_folder, silent=True)

        output = []
        for t in self._stages[0]:
            try:
                task_output_file = self._task_output_paths[t]
                local_file = self.get_path_join_method()(
                    local_folder,
                    _os.path.split(task_output_file)[1])

                unpickler = gl_pickle.GLUnpickler(local_file)
                output.append(unpickler.load())
            except Exception as e:
                if not _silent:
                    __LOGGER__.warning("Ignored exception when retrieving result for task %s, error: %s" % (t.name, e))
                output.append(None)

        # Alert --cannot remove the temp result folder because the result SFrame
        # may depend on the files to exist on disk

        return output

    def get_results(self):
        """
        Retrieve the results of the job execution. For remotely running jobs,
        this call is blocking i.e it will wait until the job completes before
        returning the results. Returns `None` if exceptions are raised.

        Returns
        -------
        result
            Result(s) of the job.

        See Also
        ---------
        get_status

        Examples
        --------
        >>> print job.get_results()
        10
        """
        # cached results to avoid retrieving again
        if hasattr(self, '_results'):
            return self._results

        is_map_job = isinstance(self._final_stage, list)
        is_map_combiner_job = len(self._stages) == 2

        if not self._job_finished():
            __LOGGER__.info("Waiting for job to finish, this may take quite a while.")
            __LOGGER__.info("You may CTRL-C to stop this command and it will not cancel your job.")
            if is_map_job:
                __LOGGER__.info("To retrieve partial results from the map job while it is running, please use get_map_results()")
            self._wait_for_job_finish()

        status = self.get_status(_silent=True)
        if status == 'Canceled':
            raise RuntimeError("The job execution was cancelled by the user. "
                               "Cannot retrieve results.")
        elif status != 'Completed':
            if is_map_combiner_job:
                raise RuntimeError("The combiner failed. To retrieve partial results "
                                   "from the map job, please use get_map_results()")
            # if is_map_job and the status is "Failed", no partial results can be
            # retrieved, so we don't suggest any further actions.
            else:
                raise RuntimeError("The job execution failed. Cannot retrieve results.")

        # status should be "Completed" at this point
        if is_map_job:
            any_task_failed = any(metric['status'] != 'Completed' for metric in self.get_metrics())
            if any_task_failed:
                raise RuntimeError("Some tasks in the map job failed. To retrieve partial "
                                   "results from the map job, please use get_map_results()")
            else:
                self._results = self._get_map_job_results()
        else:
                self._results = self._deserialize_output(self._final_stage)

        return self._results

    def get_metrics(self):
        """
        Returns live-metrics information about execution. Metrics are updated
        as the job is running. This call is not blocking i.e, it does not wait
        for the job to complete.

        Returns
        -------
        out : SFrame
            The metrics about this Job stored in an SFrame.

        See Also
        --------
        get_status

        Examples
        --------
        .. sourcecode:: python

          # for a job that ran without exceptions
          >>> job.get_metrics()
          Columns:
              task_name       str
              status  str
              start_time      int
              run_time        float
              exception_message       str
              exception_traceback     str

          Rows: 2

          Data:
          +-----------+-----------+------------+-------------------+-------------------+---------------------+
          | task_name |   status  | start_time |      run_time     | exception_message | exception_traceback |
          +-----------+-----------+------------+-------------------+-------------------+---------------------+
          |  add-0-0  | Completed | 1427930778 | 3.69548797607e-05 |                   |                     |
          |  add-1-0  | Completed | 1427930778 | 1.38282775879e-05 |                   |                     |
          +-----------+-----------+------------+-------------------+-------------------+---------------------+
          [2 rows x 6 columns]

          # for a job that threw exceptions
          >>> job.get_metrics()
          Columns:
              task_name   str
              status  str
              start_time  int
              run_time    float
              exception_message   str
              exception_traceback str

          Rows: 2

          Data:
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
        """
        if hasattr(self, '_metrics'):
            metrics = self._metrics
        else:
            metrics = self._get_metrics()

        if metrics:
            ret = _gl.SArray(metrics).unpack(column_name_prefix = '')
            ret['start_time'] = ret['start_time'].apply(lambda x: str(_datetime.datetime.fromtimestamp(x)))

            # sort output in the right order
            return ret[['task_name', 'status', 'start_time', 'run_time',\
                     'exception', 'exception_message', 'exception_traceback']]
        else:
            return None

    def _is_final_state(self, status):
        '''
        Returns whether or not the given job state means the job has finished,
        either successfully or unsucessfully
        '''
        return status in ['Failed', 'Canceled', 'Completed']

    def _job_finished(self):
        '''
        Returns whether or not the job has finished
        '''
        return self._is_final_state(self.get_status(_silent = True))

    def _wait_for_job_finish(self):
        '''
        Wait for the job to reach final state
        '''
        while not self._job_finished():
            _time.sleep(1)

    def get_start_time(self):
        """
        Returns the start time of this execution

        Returns
        -------
        out : Datetime
            Datetime representation of the start time
        """
        if self._start_time is not None:
            return _datetime.datetime.fromtimestamp(self._start_time)
        else:
            return None

    def get_end_time(self):
        """
        Returns the end time of this execution

        Returns
        -------
        out : Datetime
            Datetime representation of the start time
        """
        if self._end_time is not None:
            return _datetime.datetime.fromtimestamp(self._end_time)
        else:
            return None

    def get_log_file_path(self):
        """
        Returns the path to the execution log file of the job

        Returns
        -------
        A string representation of execution log file path
        """
        log_info_path = self.get_path_join_method()(self._exec_dir,'logs')
        return log_info_path

    def _get_exec_dir(self):
        """
        Returns a path for temporary working directories.

        Returns
        -------
        A string representation of execution dir.
        """
        if hasattr(self, '_exec_dir'):
            return self._exec_dir
        return None

    def cancel(self):
        """
        Tries to cancel the execution of the job.

        Depend on the timing, it is possible the job is already actually finished when
        the Cancel command is sent. Check get_status() to get the actual job status.

        See Also
        ---------
        get_status

        """
        if self._job_finished():
            __LOGGER__.info("Job already finished, cancel operation is not performed.")
            return

        # Do environment specific cancelation of the job
        __LOGGER__.info('Sending cancel signal to the running job.')
        try:
            self._cancel()
        except:
            __LOGGER__.error("Unable to cancel job.")

        if not self._job_finished():
            self._status = 'Canceled'
            self._finalize()

    def get_required_packages(self):
        """
        Retrieve the set of required packages for this job to execute.

        Returns
        -------
        out : set
            Set of packages required by the job for execution.

        See Also
        ---------
        get_status
        """
        return self._packages

    @staticmethod
    def _deserialize(file_path):
        """
        Takes a path to a serialized job file. Returns the deserialized
        job object.
        """
        unpickler = gl_pickle.GLUnpickler(file_path)
        ret = unpickler.load()
        unpickler.close()
        return ret

    def _serialize(self, file_path):
        """
        Serializes the Job to the provided file_path.
        """
        pickler = gl_pickle.GLPickler(file_path)
        pickler.dump(self)
        pickler.close()


    def show(self):
        """
        Visualize the Job with GraphLab Canvas. This function starts Canvas
        if it is not already running.

        Returns
        -------
        view: graphlab.canvas.view.View
            An object representing the GraphLab Canvas view

        See Also
        --------
        canvas
        """
        _canvas.get_target().state.set_selected_variable(('Jobs', self.name))
        return _canvas.show()

    def get_code(self):
        """
        Returns the code that was used to execute this job.

        Returns
        -------
        out : str
            The string representation of the code that was used to run this job.
        """
        return self._stages[0][0].get_code()

    def get_parameters(self):
        """
        Returns the parameters, if any, that was used to run this job.

        Returns
        -------
        out : dictionary
            The dictionary containing the parameters that was used to run this job.
        """
        parameters = dict()
        for task in self._stages:
            for t in task:
                parameters[t.get_name()] = t.get_inputs()
        return parameters

    def _get_task_names(self):
        names = []
        for task in self._stages:
            if isinstance(task, list):
                names += [t.name for t in task]
            elif hasattr(task, 'name'):
                names.append(task.name)
            else:
                __LOGGER__.warning("Unable to get task name for '%s' task. Skipping")
        return names

    def _get_help_str(self):
        header =     "Help\n------"
        show =       "Visualize progress : self.show()"
        status =     "Query status       : self.get_status()"
        results =    "Get results        : self.get_results()"
        return "\n".join([header, show, status, results])

    def _get_str(self):
        status = self.get_status(_silent=True)
        task_names = self._get_task_names()
        if len(str(task_names)) > 80:
            if self._num_tasks == 1:
                task_names = str(task_names)[:75] + " ... ]"
            else:
                task_names = "%s, %s, %s ... (total %s functions)." % (task_names[0],
                       task_names[1], task_names[2], self._num_tasks)

        # Call metrics here would also cause status to be checked, so avoid
        # calling get_status again below by directly getting self._status
        metrics  = self.get_metrics().__str__()

        header   = "Info\n------"
        job      = "Job                : %s" % self.name
        func     = "Function(s)        : %s" % task_names
        status   = "Status             : %s" % status

        if self._error:
            error  = "Error              : %s" % self._error

        env_head = "Environment\n----------"

        met_head = "Metrics\n-------"
        job_start_time  = "Start time         : %s" % self.get_start_time()
        job_end_time    = "End time           : %s" % self.get_end_time()

        out = "\n".join([header, job, func, status])
        if self._error:
            out += "\n%s" % error

        out += "\n\n%s" % self._get_help_str()
        out += "\n\n%s" % "\n".join([env_head, self.environment.__str__()])
        out += "\n\n%s" % "\n".join([met_head, job_start_time, job_end_time, metrics])
        return out

    def _finalize(self):
        '''
        When the job finishes, query the metrics and task status one last
        time, and save. The job cannot be saved after that.
        '''
        # If a job is canceled, then metrics file may not be in a valid state
        # do not try to read the metrics.
        if self._status != 'Canceled':
            self._metrics = self._get_metrics()
        else:
            self._metrics = []
            status_path = self.get_path_join_method()(self._exec_dir, 'status')
            if _file_util.is_local_path(status_path):
                try:
                    with open(status_path, 'w') as f:
                        _json.dump({'status': 'Canceled', 'start_time':None, 'end_time':None}, f)
                except Exception as e:
                    _logging.info('Exception trying to write job info')
                    _logging.info(e)
                    pass

        # Fail the job if last stage contains one single task. This could be
        # a single task job, or a map job with combiner.
        if not isinstance(self._final_stage, list) and \
                                       self._status == 'Completed' and \
                                       len(self._metrics) > 0 and \
                                       self._metrics[-1]['status'] == 'Failed':
            self._status = 'Failed'

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self.__str__()

    @classmethod
    def _resolve_backward_compatibility(cls, old_version, new_version):
        '''
        Default way of resolving backward compatibility for job

        For now, GLC 1.4 is not compatible with 1.3 and 1.3 is not compatible
        with previous version.
        '''
        assert(old_version <= new_version)

        if old_version == 1:
            raise RuntimeError("Cannot load job with version 1 schema, please use "
                "GraphLab Create v1.3 to read the job file instead. If you do not "
                "care about current session objects any more, you may simply remove "
                "all old objects by deleting directory: '~/.graphlab/artifacts.'")

class LocalAsynchronousJob(Job):

    # See JOB_VERSION comments above, version 2 is not compatible with version 1
    _LOCAL_ASYNC_JOB_VERSION = 2

    def __init__(self, pid, job):
        super(LocalAsynchronousJob, self).__init__(
            name = job.name,
            stages = job._stages,
            final_stage = job._final_stage,
            environment = job.environment,
            _exec_dir = job._exec_dir)

        self._pid = pid
        self._task_output_paths = job._task_output_paths

    def _get_version(self):
        return self._LOCAL_ASYNC_JOB_VERSION

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : A GLUnpickler archive.
        """
        # Load the dump.
        obj = unpickler.load()

        # Construct a new object.
        new_job = Job(obj.name, obj._stages, obj._final_stage, obj.environment)
        new = cls(obj._pid, new_job)

        cls._resolve_backward_compatibility(version, new._get_version())

        # Now copy over the useful parts of the dump.
        lst = ['_exec_dir', '_final_stage', '_pid', '_task_output_paths',
               '_status', '_task_status', '_typename',
               'environment', '_metrics', 'name', '_num_tasks',
               '_packages', '_stages', '_start_time', '_end_time']
        _internal_utils.copy_attributes(new, obj, lst)
        return new

    def __str__(self):
        status = self.get_status()
        result = self._get_str()
        result += "\n\nExecution Information\n---------------------\n"
        if status not in ['Completed', 'Failed', 'Canceled']:
            result += "Process pid          : %d\n" % self._pid
        result +=     "Execution Directory  : %s\n" % self._get_exec_dir()
        result +=     "Log file             : %s\n" % self.get_log_file_path()
        return result

    def _cancel(self):
        if _psutil.pid_is_running(self._pid):
            _psutil.kill_process(self._pid)

    def get_log_file_path(self):
        """
        Returns log file path directory that contains all the workers logs in the current execution for HadoopJobs.

        Returns
        -------
        out : str
            The string representation of path to the log file directory.

        Note: Call job.flush_logs() if log file path directory is empty.

        """
        log_info_path = _os.path.join(self._exec_dir,'execution.log')
        return log_info_path

class Ec2Job(Job):

    @classmethod
    def get_path_join_method(cls):
        return lambda *args : '/'.join(args)

    # See JOB_VERSION comments above, version 2 is not compatible with version 1
    _EC2_JOB_VERSION = 2

    def __init__(self, job, app_id):
        assert app_id is not None

        super(Ec2Job, self).__init__(
            job.name,
            stages = job._stages,
            final_stage = job._final_stage,
            environment = job.environment,
            _task_output_paths = job._task_output_paths,
            _exec_dir = job._exec_dir,
            _job_type = job._job_type)

        self._app_id = app_id
        self._s3_log_url = None


    def _get_version(self):
        return self._EC2_JOB_VERSION

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : A GLUnpickler archive.
        """
        # Load the dump.
        obj = unpickler.load()

        # Construct a new object.
        new_job = Job(obj.name, stages=obj._stages, final_stage=obj._final_stage,
                      environment=obj.environment, _exec_dir=obj._exec_dir,
                      _task_output_paths=obj._task_output_paths,
                      _job_type = obj._job_type)
        new = cls(new_job, obj._app_id)
        cls._resolve_backward_compatibility(version, new._get_version())

        # Now copy over the useful parts of the dump.
        lst = ['_status', '_metrics', '_start_time', '_end_time',
               '_s3_log_url']
        _internal_utils.copy_attributes(new, obj, lst)

        return new

    def _cancel(self):
        """
        Cancels this job. This will terminate all associated EC2 instances.

        Returns
        -------
        True if successfully canceled, False otherwise.

        See Also
        ---------
        get_status
        """
        self.environment._cancel_job(self._app_id)

    def __str__(self):
        result = self._get_str()
        result += "\n\nExecution Information:\n\n"
        result += "  EC2 Environment name: %s\n" % self.environment.name
        result += "  Log Path: %s\n" % self.get_log_file_path()

        return result


class HadoopJob(Job):

    # See JOB_VERSION comments above, version 2 is not compatible with version 1
    _HADOOP_JOB_VERSION = 2
    @classmethod
    def get_path_join_method(cls):
        return lambda *args : '/'.join(args)


    # need hadoop and yarn on the command line
    # need if source = none need virtualenv on the commandline and python
    def __init__(self, job, app_id):
        assert app_id is not None
        super(HadoopJob, self).__init__(
            job.name,
            stages=job._stages,
            final_stage = job._final_stage,
            environment = job.environment,
            _task_output_paths = job._task_output_paths,
            _exec_dir = job._exec_dir,
            _job_type = job._job_type)

        self.app_id = app_id
        self._app_state = None
        self._yarn_AM_state = None
        self._yarn_end_time = None
        self._yarn_start_time = None

    def _get_version(self):
        return self._HADOOP_JOB_VERSION

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : A GLUnpickler archive.
        """
        # Load the dump.
        obj = unpickler.load()

        # Construct a new object.
        new_job = Job(obj.name, stages=obj._stages, final_stage=obj._final_stage,
                      environment=obj.environment, _exec_dir=obj._exec_dir,
                      _task_output_paths=obj._task_output_paths,
                      _job_type = obj._job_type)
        new = cls(new_job, obj.app_id)
        cls._resolve_backward_compatibility(version, new._get_version())

        lst = ['_exec_dir', '_final_stage', '_pid', '_task_output_paths',
               '_status', '_task_status', '_typename',
               'environment', '_metrics', 'name', '_num_tasks',
               '_packages', '_stages', '_app_state', '_end_time', '_start_time',
               '_yarn_AM_state', 'app_id', '_yarn_start_time', '_yarn_end_time']

        # Now copy over the useful parts of the dump.
        _internal_utils.copy_attributes(new, obj, lst)
        return new

    def flush_logs(self):
        """
        Force flushing the logs in the commander and workers for HadoopJobs
        """
        self._remote_flush_logs()

    def get_error(self):
        """
        Return error message from HadoopJobs
        """
        status = self.get_status()
        if status != 'Failed':
            return None
        else:
            return self._error

    def _get_status(self, _silent = False):
        """
        Get job status for the Hadoop job.

        We try to consolidate state information from both Yarn and our status
        file.
            1. If our status file indicates job is in fianl state, use that state.
            2. If Yarn application status is Failed or Unknown or our status file
               do not exist, use yarn application state.
            3. Otherwise, use our state
        """
        # Get our job status file first
        status_file = self._exec_dir + '/status'
        status = self._load_file_and_parse(status_file, self._parse_status_file, silent = _silent, test_url=False)

        # status file should never be none now
        if self._is_final_state(status):
            return status

        # Mapping from Yarn application state to our job state
        yarn_state_to_job_state = {
            'KILLED': 'Canceled',
            'UNDEFINED': 'Pending',
            'FAILED' : 'Failed',
            'SUCCEEDED': 'Completed',
            'UNKNOWN': 'Unknown'
        }

        # Get YARN application state
        from ._executionenvironment import HadoopExecutionEnvironment as _HadoopExecutionEnvironment
        yarn_app_states = _HadoopExecutionEnvironment.get_yarn_application_state(
            self.environment,
            self.app_id,
            silent = True)

        yarn_state = yarn_app_states['DistributedFinalState'] if yarn_app_states else 'UNKNOWN'

        # Consolidate the result
        if yarn_state == 'FAILED' or yarn_state == 'UNKNOWN' or (status is None):
            # get status from <appname>/<appid>/cmd_exec_dir/status
            user_home_dir = _HadoopExecutionEnvironment._get_user_hdfs_home_dir(self.environment)
            am_status_file = '%s/%s/%s/cmd_exec_dir/status' % (user_home_dir, 'turi_distributed', self.app_id)
            self._load_file_and_parse(am_status_file, self._parse_status_file, silent = _silent, test_url=True)

            return yarn_state_to_job_state.get(yarn_state)
        else:
            return status

    def _cancel(self):
        from ._executionenvironment import HadoopExecutionEnvironment as _HadoopExecutionEnvironment
        success = _HadoopExecutionEnvironment.cancel_yarn_application(
            self.environment, self.app_id, silent = False)

        if not success:
            self._status= 'Unknown'

    def __str__(self):
        result = self._get_str()
        result += "\nApplication Id: %s\n" % self.app_id

        if self.environment.hadoop_conf_dir:
            result += "Command to view yarn logs: \n "\
                      "yarn --config %s logs -applicationId %s" % (\
                              self.environment.hadoop_conf_dir, self.app_id)
        else:
            result += "Command to view yarn logs: \n  " \
                      "yarn logs -applicationId %s" % self.app_id

        return result

class DmlJob(Job):
    """
    A representation of DML job.

    DML job is like the other jobs, the only difference is that it is submitted
    to a long running "DML Execution Environment". So the cancelation of the job
    goes through the 'commander' in the "DML Execution Environment", not going
    through specific environment (EC2 or Hadoop or LocalAsync)
    """
    @classmethod
    def get_path_join_method(cls):
        return lambda *args : '/'.join(args)

    def __init__(self, job):
        self.dml_cluster = job.environment
        base_environment = job.environment.environment
        super(DmlJob, self).__init__(
            name = job.name,
            stages=job._stages,
            final_stage = job._final_stage,
            environment = base_environment,
            _task_output_paths = job._task_output_paths,
            _exec_dir = job._exec_dir,
            _job_type = 'DML')

    def _cancel(self):
        return self.dml_cluster.cancel_job()

    def _get_status(self, _silent = False):
        if not self.dml_cluster.get_status(_silent):
            __LOGGER__.info('Cannot connect to DML execution engine, mark job as failed.')
            return 'Failed'

        return super(DmlJob, self)._get_status(_silent)

    def __str__(self):
        result = self._get_str()
        result += "DML Cluster: %s\n" % self.dml_cluster.__str__()

        return result
