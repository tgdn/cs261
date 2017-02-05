from copy import copy as _copy
import json as _json
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import time as _time
import sys as _sys
import logging
import traceback

import six as _six
import requests as _requests
from uuid import uuid4 as _uuid

from . import _timer
import graphlab as _gl
import graphlab.connect as _mt
from graphlab.connect.aws._ec2 import _ec2_factory, _ProductType
from graphlab.util import _make_temp_directory
from graphlab import _gl_pickle
from . import _job

from ._task import Task as _Task
from . import _internal_utils
from graphlab.util import file_util as _file_util

__LOGGER__ = logging.getLogger(__name__)

def _get_execution_env(environment):
    """
    Returns an execution environment for the corresponding env.

    Parameters
    ----------
    environment : Environment
        Environment that this job is running on.

    num_tasks :
        Number of tasks (for metric tracking)

    Returns
    -------
    An ExecutionEnvironment corresponding to the env.
    """
    if environment is None:
        typename = 'LocalAsync'
    else:
        typename = type(environment).__name__

    tracker = _mt._get_metric_tracker()
    tracker.track('deploy.job.create.%s' % typename.lower(), value=1)

    if typename == 'Local':
        exec_env = LocalExecutionEnvironment
    elif typename == 'LocalAsync':
        exec_env = LocalAsynchronousEnvironment
    elif typename in ['EC2', 'Ec2Cluster']:
        exec_env = Ec2ExecutionEnvironment
    elif typename in ['Hadoop', 'HadoopCluster']:
        exec_env = HadoopExecutionEnvironment
    else:
        raise Exception("Validation Failed: Unknown execution environment.")

    return exec_env


class ExecutionEnvironment(object):
    """
    Base class for all environments. Each derived class is expected to implement
    the run_job method which runs a job.
    """

    @staticmethod
    def prepare_job_exec_dir(job, destination):
        '''
        Write job initial state, metrics, job definition to the destination folder
        to prepare for job execution.

        The prep work includes:
        * serialize job definition (pickled job)
        * json serialized job metadata
        * initial status file
        * initial metrics file
        * initial commander init file
        * initial folder for logs and output
        '''
        if not _os.path.isdir(destination):
            raise RuntimeError("'%s' has to be a local directory." % destination)

        __LOGGER__.debug('Preparing all job files.')

        # write job definition
        job_definition_path = _os.path.join(destination, 'job-definition')
        _job.Job._serialize(job, job_definition_path)

        # write job metadata
        metadata_file_path = _os.path.join(destination, 'metadata')
        ExecutionEnvironment.write_job_metadata(job, metadata_file_path)

        # write initial status file
        state_file_path = _os.path.join(destination, 'status')
        status = {'status':'Pending', 'error':None, 'start_time':None, 'end_time':None}
        with open(state_file_path, 'w') as f:
            _json.dump(status, f)

        # write initial metrics file
        metrics_file_path = _os.path.join(destination, 'metrics')
        with open(metrics_file_path, 'w') as f:
            _json.dump({}, f)

        # write commander init file
        commander_status_path = _os.path.join(destination, 'commander_init.status')
        ExecutionEnvironment._write_commander_init_file(commander_status_path)

        _os.mkdir(_os.path.join(destination, 'logs'))
        _os.mkdir(_os.path.join(destination, 'output'))

    @staticmethod
    def write_required_packages(path, additional_packages):
        '''
        Write all user's python dependent packages into a file called
            user_requirements.pip
        '''
        reqs_pip = _os.path.join(path, "user_requirements.pip")
        reqs = additional_packages
        if not reqs or len(reqs) == 0:
            return None

        with open(reqs_pip, 'w') as f:
            for line in reqs:
                f.write(line +'\n')

        return reqs_pip

    @staticmethod
    def _write_commander_init_file(path):
        with open(path, 'w') as f:
            _json.dump({'host_name': None, 'port':-1}, f)

    @staticmethod
    def write_job_metadata(job, path):
        '''
        Write metadata file for the job
        '''
        stages = job._stages
        new_stages = [[(t.name) for t in s] for s in stages]

        metadata = {
            "name"          : job.name,
            "working_dir"   : job._exec_dir,
            "job_type"      : job._job_type,
            "stages"        : new_stages,
            'timeout_in_seconds': 0
        }

        with open(path, 'w') as meta_file:
            _json.dump(metadata, meta_file)

class LocalExecutionEnvironment(ExecutionEnvironment):
    """
    An ExecutionEnvironment for local, synchronous execution.
    """
    @staticmethod
    def run_job(job):
        job._status = 'Running'

        try:
            job._start_time = int(_time.time())

            __LOGGER__.info("Job execution started: %s, execution path: %s" % (job.name, job._exec_dir))

            # Nothing done in parallel.
            job._metrics = []

            for cur_step in job._stages:
                for task in cur_step:
                    task_metrics = LocalExecutionEnvironment._run_task(task, job, job._task_output_paths[task])
                    job._metrics.append(task_metrics)
                    job._task_status[task.name] = task_metrics['status']

            __LOGGER__.info("Execution completed : %s" % job.name)

            # Fail the job if last stage contains one single task. This could be
            # a single task job, or a map job with combiner.
            if not isinstance(job._final_stage, list) and \
                job._metrics[-1]['status'] == 'Failed':

                job._status = 'Failed'
                job._error = 'Exception encounted when running job, check job.get_metrics() for more information'
            else:
                job._status = 'Completed'
                job._error = None

        except Exception as e:
            trace = traceback.format_exc()
            err_msg = "Job execution failed.\n"
            err_msg += "Traceback (most recent call last)\n %s\n" % trace
            err_msg += "Error type    : %s\n" % e.__class__.__name__
            err_msg += "Error message : %s\n" % str(e)
            __LOGGER__.error(err_msg)

            job._status = 'Failed'
            job._error = err_msg

        job._end_time = int(_time.time())

        return job

    @staticmethod
    def _run_task(task, job, output_path):
        '''
        Runs given task locally, write output to output_path
        '''
        start_time = int(_time.time())
        result = None

        ret = {
            'task_name': task.name,
            'status': 'Running',
            'output_path': output_path,
            'start_time': start_time,
            'run_time': None,
            'exception': None,
            'exception_message': None,
            'exception_traceback': None
        }

        try:
            __LOGGER__.info("Task started: %s, output path: %s" % (task.name, output_path))

            with _timer.Timer() as t:
                code = task.get_runnable_code()

                ## resolve task input
                inputs = task.get_inputs()
                for k, v in inputs.items():
                    if isinstance(v,_Task):
                        inputs[k] = job._deserialize_output(v)
                        __LOGGER__.info("task input[k]: %s" %inputs[k])

                # run the task code
                result = code(**inputs)

            ret['run_time'] = t.secs
            ret['status'] = 'Completed'

            # Log completion.
            __LOGGER__.info("Task completed: %s" % task.name)

        # Task Failed
        except Exception as e:
            trace = traceback.format_exc()
            ret['exception'] = e.__class__.__name__
            ret['exception_message'] = str(e)
            ret['exception_traceback'] = trace
            ret['status'] = 'Failed'

            # Log error message.
            err_msg = "Task execution failed.\n"
            err_msg += "Traceback (most recent call last)\n %s\n" % trace
            err_msg += "Error type    : %s\n" % e.__class__.__name__
            err_msg += "Error message : %s\n" % str(e)
            __LOGGER__.info(err_msg)


        # persist output in case of failure too to make it easy to process
        # partial failures in map job case
        LocalExecutionEnvironment._persist_task_output(result, output_path)

        return ret

    @staticmethod
    def _persist_task_output(output, output_path):
        '''
        Persist task output to output path, the path can be either local or remote
        '''
        pickler = _gl_pickle.GLPickler(output_path)
        pickler.dump(output)
        pickler.close()


class Ec2ExecutionEnvironment(ExecutionEnvironment):
    """
    An ExecutionEnvironment for deploying jobs in EC2.
    """
    @staticmethod
    @_file_util.retry(tries=240, delay=2, retry_exception=_requests.exceptions.ConnectionError)
    def _get_package_list(host):
        response = _requests.get("http://%s:9004/list_packages" % host)
        return response.json()['packages']

    @staticmethod
    @_file_util.retry(tries=240, delay=2, retry_exception=_requests.exceptions.ConnectionError)
    def _make_http_post_request(host_name, path, parameters={}):
        # Swallow all logging from the 'requests' module.
        logging.getLogger('requests').setLevel(logging.CRITICAL)

        response = _requests.post("http://%s:9004/%s" % (host_name, path), params = parameters)

        # Check the response
        if not response:
            error_msg = "Server side error"
            if response is not None:
                error_msg += ": %s" % response.json()
            raise RuntimeError(error_msg)

        return response

    @staticmethod
    def _is_host_pingable(host):
        try:
            response = _requests.get("http://%s:9004/ping" % host, timeout = 5)
        except:
            return False
        return bool(response)

    @staticmethod
    def _start_commander_host(env_name, config, s3_folder_path, num_hosts, additional_packages,
                              idle_shutdown_timeout):
        @_file_util.retry(tries=240, delay=2, retry_exception=_requests.exceptions.ConnectionError)
        def _wait_for_host_to_start_up():
            response = _requests.get("http://%s:9004/ping" % commander.public_dns_name)
            if not response:
                raise RuntimeError()

        credentials = config.get_credentials()

        # Set user data for cluster controller
        num_children = num_hosts - 1
        user_data = {
            'auth_token': '', 'AWS_ACCESS_KEY_ID': credentials['aws_access_key_id'],
            'AWS_SECRET_ACCESS_KEY': credentials['aws_secret_access_key'], 'daemon': True,
            'is_cluster_controller': True, 'num_children_host': num_children,
            's3_folder_path': s3_folder_path, 'additional_packages': additional_packages,
            'idle_shutdown_timeout': idle_shutdown_timeout
            }

        # Propagating debug environment variables to user data
        if('GRAPHLAB_TEST_AMI_ID' in _os.environ and 'GRAPHLAB_TEST_ENGINE_URL' in _os.environ
           and 'GRAPHLAB_TEST_OS_URL' in _os.environ and 'GRAPHLAB_TEST_HASH_KEY' in _os.environ):
            user_data['GRAPHLAB_TEST_AMI_ID'] = _os.environ['GRAPHLAB_TEST_AMI_ID']
            user_data['GRAPHLAB_TEST_ENGINE_URL'] = _os.environ['GRAPHLAB_TEST_ENGINE_URL']
            user_data['GRAPHLAB_TEST_OS_URL'] = _os.environ['GRAPHLAB_TEST_OS_URL']
            user_data['GRAPHLAB_TEST_HASH_KEY'] = _os.environ['GRAPHLAB_TEST_HASH_KEY']
        if('GRAPHLAB_TEST_EC2_KEYPAIR' in _os.environ):
            user_data['GRAPHLAB_TEST_EC2_KEYPAIR'] = _os.environ['GRAPHLAB_TEST_EC2_KEYPAIR']

        # Launch the cluster controller
        tags = _copy(config.tags)
        tags['Name'] = env_name
        commander, security_group, subnet_id = _ec2_factory(config.instance_type, region = config.region,
                                CIDR_rule = config.cidr_ip,
                                security_group_name = config.security_group,
                                tags = tags, user_data = user_data,
                                credentials = credentials,
                                product_type = _ProductType.TuriDistributed,
                                subnet_id = config.subnet_id,
                                security_group_id = config.security_group_id)

        # Log message explaining the additional hosts will not be launched by us.
        if num_children == 1:
            __LOGGER__.info("One additional host will be launched by %s" % commander.instance_id)
        elif num_children > 1:
            __LOGGER__.info("%d additional hosts will be launched by %s"
                            % (num_children, commander.instance_id))

        # Wait for cluster_controller_daemon
        __LOGGER__.info("Waiting for %s to start up." % commander.instance_id)
        try:
            _wait_for_host_to_start_up()
        except:
            raise RuntimeError('Unable to start host(s). Please terminate '
                               'manually from the AWS console.')

        return commander.public_dns_name

    @staticmethod
    def _stop_cluster(cluster_controller_dns):
        __LOGGER__.info('Stopping cluster')
        try:
            Ec2ExecutionEnvironment._make_http_post_request(cluster_controller_dns, 'terminate')
        except:
            raise RuntimeError('Unable to terminate host(s). Please terminate manually'
                               ' from the AWS console.')

    @staticmethod
    def get_job_state(environment, app_id):
        try:
            response = _requests.get("http://%s:9004/get_status" % environment.cluster_controller,
                                        params = {'app_id': app_id})

            return response.json()

        except Exception as e:
            raise RuntimeError('Unable to get job status from EC2 cluster. Error: %s.' % e)

    @staticmethod
    def run_job(job):
        '''
        Run a PIPELINE job
        '''
        assert job._job_type == 'PIPELINE'

        environment = job.environment
        if not environment.cluster_controller:
            raise RuntimeError('This environment must be started before you can use it to create jobs.' \
                                   ' Call \'start\' to start the environment.')

        Ec2ExecutionEnvironment.prepare_job_files(environment, job)

        __LOGGER__.info('Submitting job...')

        # Post the job request to the cluster controller
        app_id = Ec2ExecutionEnvironment.submit_job(
            job.environment,
            job._get_exec_dir(),
            job._get_max_concurrent_tasks())

        __LOGGER__.info('Job submitted successfully.')

        # Create the job that will be returned
        return  _job.Ec2Job(job, app_id)

    @staticmethod
    def prepare_job_files(environment, job):
        '''
        Prepare all files needed to run a job in EC2 cluster
        '''

        # Update job working directory
        s3_job_path_root = Ec2ExecutionEnvironment.create_job_home_dir(environment, job.name)
        _job.Ec2Job._update_exec_dir(job, s3_job_path_root)

        __LOGGER__.info('Job working directory: %s' % s3_job_path_root)

        # Prepare all files locally and then upload to S3
        temp_local_folder = _make_temp_directory(prefix='ec2_job_')
        try:
            ExecutionEnvironment.prepare_job_exec_dir(job,  temp_local_folder)

            _file_util.upload_to_s3(
                temp_local_folder,
                s3_job_path_root,
                is_dir = True,
                aws_credentials = environment.ec2_config.get_credentials(),
                silent = True)
        finally:
                _shutil.rmtree(temp_local_folder)

    @staticmethod
    def create_job_home_dir(environment, job_name):
        '''
        Given a job name, create a home directory for the job in EC2 cluster
        '''
        return environment.s3_state_path + '/' + job_name + '-' + str(_uuid())

    @staticmethod
    def submit_job(environment, job_working_dir, max_concurrent_tasks, silent = False):
        '''
        Submit a job to EC2 cluster
        '''
        try:
            post_parms = {'s3_job_root_path': job_working_dir,
                          'max_concurrent_tasks': max_concurrent_tasks}

            response = Ec2ExecutionEnvironment._make_http_post_request(
                environment.cluster_controller,'submit',parameters=post_parms)
            if not response:
                raise RuntimeError('Unexpected response from submitting job to '
                    'EC2 cluster: %s' % response.text)

            return response.json()['app_id']
        except Exception as e:
            __LOGGER__.error('Error createing DML execution engine in EC2 cluster: %s' % e)
            raise

    @staticmethod
    def cancel_job(environment, app_id, silent = False):
        '''
        Cancel a job with given app_id
        '''
        cancel_url = "http://%s:9004/cancel" % environment.cluster_controller
        try:
            response = _requests.post(cancel_url, params = {'app_id': app_id})
            if not response:
                raise RuntimeError("Cannot cancel job, error: %s" % response.text)
            return True
        except Exception as e:
            __LOGGER__.error("Unable to cancel job with id %s, error: %s." % (app_id, e))
            return False

class HadoopExecutionEnvironment(ExecutionEnvironment):
    """
    An ExecutionEnvironment for Hadoop jobs.
    """
    _json_flag = '$$JSONTAG##'

    @staticmethod
    def run_job(job):
        assert job._job_type == 'PIPELINE'

        __LOGGER__.info('Submtting job...')

        HadoopExecutionEnvironment.prepare_job_files(job.environment, job)

        app_id = HadoopExecutionEnvironment.submit_job(job.environment, job._exec_dir)

        __LOGGER__.info('Job submitted successfully.')

        # Create hadoop job instance
        return _job.HadoopJob(job, app_id)

    @staticmethod
    def prepare_job_files(environment, job):
        '''
        Upload all job related information to HDFS so that it can be executed
        remotely
        '''
        exec_dir = HadoopExecutionEnvironment.create_job_home_dir(environment, job.name)
        _job.HadoopJob._update_exec_dir(job, exec_dir)

        logging.info("Job working directory: %s" % job._exec_dir)

        temp_job_folder = _make_temp_directory(prefix='hadoop_job_')
        try:
            ExecutionEnvironment.prepare_job_exec_dir(job, temp_job_folder)

            # Move everything to HDFS
            _file_util.upload_folder_to_hdfs(
                temp_job_folder,
                exec_dir,
                hadoop_conf_dir = environment.hadoop_conf_dir)

        finally:
            _shutil.rmtree(temp_job_folder)

    @staticmethod
    def submit_job(environment, job_working_dir, silent = False):
        '''
        Submit a new job to Hadoop
        '''
        app_info =  HadoopExecutionEnvironment.submit_yarn_application(
            environment,
            job_working_dir = job_working_dir,
            silent = silent)
        return app_info['app_id']


    @staticmethod
    def create_job_home_dir(environment, job_name):
        '''
        Given a job name, create a home directory for the job in Hadoop cluster
        '''
        user_home_dir = HadoopExecutionEnvironment._get_user_hdfs_home_dir(environment)
        job_home = "%s/turi_distributed/jobs" % user_home_dir

        job_working_dir = '%s/turi_distributed/jobs/%s' % (user_home_dir, job_name)

        _file_util.hdfs_mkdir(job_home, environment.hadoop_conf_dir)

        return job_working_dir

    @staticmethod
    def _get_user_hdfs_home_dir(environment):
        '''
        Given Hadoop environment information, find current user's HDFS home directory
        '''
        configs = _internal_utils.find_core_hadoop_config(environment.hadoop_conf_dir)
        if 'HADOOP_USER_NAME' in _os.environ:
            user = _os.environ['HADOOP_USER_NAME']
        else:
            if _sys.platform == 'win32':
                user = _os.environ['USERNAME']
            else:
                user = _os.environ['USER']

        return 'hdfs://%s:%s/user/%s' % (configs[0]['namenode'], configs[0]['port'], user)

    @staticmethod
    def submit_yarn_application(environment, job_working_dir, silent = False,
        native_conda = False):
        '''
        Submit a yarn application
        '''
        # Create temporary directory to write tmp files in for the job
        temp_local_directory = _make_temp_directory(prefix='hadoop_job_')

        req_file_path = None
        if hasattr(environment, 'additional_packages'):
            req_file_path = ExecutionEnvironment.write_required_packages(
                temp_local_directory, environment.additional_packages)

        # build actual hadoop command
        hadoop_cmd = HadoopExecutionEnvironment._build_hadoop_cmd(
            environment,
            job_working_dir,
            req_file_path,
            native_conda = native_conda)

        # call hadoop command
        if not silent:
            __LOGGER__.info("Submitting job to Hadoop cluster using command= \n%s" % hadoop_cmd)

        proc = _subprocess.Popen(hadoop_cmd, shell=True,
                                          stderr=_subprocess.STDOUT,
                                          stdout=_subprocess.PIPE)

        app_info = HadoopExecutionEnvironment._parse_hadoop_cmd_output(proc, silent)

        if 'app_id' not in app_info or not app_info['app_id']:
            __LOGGER__.info("Submitting job to Hadoop cluster using command= \n%s" % hadoop_cmd)
            raise RuntimeError("Error submitting application or determining application id. Please confirm that you"
                               " can correctly access the Hadoop environment specified (try running the above"
                               " command from a terminal to see more diagnostic output).")

        else:
            _shutil.rmtree(temp_local_directory)

        return app_info


    @staticmethod
    def _parse_hadoop_cmd_output(proc, silent = False):
        app_id = None
        AM_state = None
        app_state = None
        start_time = None
        end_time = None


        debug = _os.environ.get('GRAPHLAB_TEST_DEBUG', False)

        for line in proc.stdout:
            if debug:
                print(line.strip('\n'))

            if HadoopExecutionEnvironment._json_flag in line:
                ## json line looks like TAGjsonTAG
                clean = line.split(HadoopExecutionEnvironment._json_flag)
                app_report = _json.loads(clean[1])
                app_id = app_report['AppId']
                AM_state = app_report['AppState']
                app_state = app_report['DistributedFinalState']
                start_time = app_report['StartTime']
                end_time = app_report['FinishTime']

                if not silent:
                    for key,value in _six.iteritems(app_report):
                        __LOGGER__.info('%s=%s' % (key, value))

        return {'app_id': app_id, 'app_state': app_state, 'AM_state': AM_state,
                'start_time': start_time, 'end_time': end_time}


    @staticmethod
    def _build_hadoop_cmd(
        environment,
        job_working_dir,
        user_reqs_file = None,
        native_conda = False):
        '''
        Build Hadoop command

        Arguments
        ----------
        environment : :py:class:`~graphlab.aws.deploy.environment._Environment`
            The environment object associated with this execution environment

        job_working_dir : str
            The job working directory
        '''

        cmd_args = {}
        cmd_args['job_working_dir'] = job_working_dir

        cmd_args['appname'] = "turi_distributed"
        cmd_args['jar'] = environment._get_hadoop_jar()  # client jar

        # Container information
        cmd_args['num_containers'] = environment.num_containers
        cmd_args['container_memory'] = environment.container_size
        cmd_args['container_vcores'] = environment.num_vcores

        # Distributed execution information
        cmd_args['commander_port_start'] = environment.start_port
        cmd_args['commander_port_end'] = environment.end_port
        cmd_args['worker_port_start'] = environment.start_port
        cmd_args['worker_port_end'] = environment.end_port
        cmd_args['turi_distrib_path'] = environment.turi_dist_path

        if native_conda:
            cmd_args['native_conda'] = native_conda

        if environment.node_tmp_dir:
            cmd_args['node_tmp_dir'] = environment.node_tmp_dir

        if environment.hdfs_tmp_dir:
            cmd_args['hdfs_tmp_dir'] = environment.hdfs_tmp_dir

        if user_reqs_file is not None:
            cmd_args['reqs_file'] = user_reqs_file

        if environment._conda_installed_dir:
            cmd_args['conda_installed_dir'] = environment._conda_installed_dir


        cmd_args['json'] = None

        # Now really build the command
        hadoop_base = 'hadoop'
        if environment.hadoop_conf_dir is not None:
            hadoop_base += " --config %s " % environment.hadoop_conf_dir

        client_jar_file = environment._get_hadoop_jar()
        hadoop_cmd = "%s jar %s" % (hadoop_base, client_jar_file)

        for k, v in cmd_args.items():
            if v is None:
                hadoop_cmd += " -%s" % k
            else:
                hadoop_cmd += " -%s %s" % (k, v)

        return hadoop_cmd

    @staticmethod
    def get_yarn_application_state(environment, app_id, silent=False):
        """
        Retrieve Yarn application state

        Parameters
        -----------
        environment : HadoopCluster
            The Hadoop cluster environment object

        app_id : str
            The Yarn application id to get state from

        silent : bool
            If True, do not print out informational message

        Returns
        --------
        app_state : dict
            a report from Yan for given application, includes the following fields:
            DistributedFinalState -- final state of the application
            AppState -- AM state of the application

            None if something goes wrong, or Yarn cannot find the given application
        """
        hadoop_base = "hadoop "
        if environment.hadoop_conf_dir:
            hadoop_base += "--config %s " % environment.hadoop_conf_dir

        hadoop_jar = environment._get_hadoop_jar()

        hadoop_cmd = "%s jar %s -jar %s -checkAppId %s -json" % \
                     (hadoop_base, hadoop_jar, hadoop_jar, app_id)

        if not silent:
            __LOGGER__.info(("Retrieving current job status from Hadoop cluster"
                                  " using command= \n %s") % hadoop_cmd)

        proc = _subprocess.Popen(hadoop_cmd, shell=True,
                                 stderr=_subprocess.STDOUT,
                                 stdout=_subprocess.PIPE)
        app_report = None
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if HadoopExecutionEnvironment._json_flag in line:
                clean = line.split(HadoopExecutionEnvironment._json_flag)
                app_report = _json.loads(clean[1])
                break

        # print the output in case something wrong talking to the yarn
        if app_report == None:
            for l in lines:
                __LOGGER__.error(l)

        return app_report

    @staticmethod
    def cancel_yarn_application(environment, app_id, silent = False):
        '''
        Cancel a Yarn application

        Parameters
        ----------
        environment : Cluster
            The cluster environment

        app_id : str
            Yarn application id to cancel

        silent : bool
            If True, do not print informational message

        Returns
        -------
        success : boolean
            True if the application is canceled successfully, False otherwise

        '''
        yarn_cmd = "yarn "
        if environment.hadoop_conf_dir:
            yarn_cmd += " --config %s " % environment.hadoop_conf_dir

        yarn_cmd += " application -kill %s " % app_id

        if not silent:
            __LOGGER__.info("Attempting to kill job using the following command: \n %s" % yarn_cmd)

        proc = _subprocess.Popen(yarn_cmd, shell=True,
                                  stderr=_subprocess.STDOUT,
                                  stdout=_subprocess.PIPE )

        message = None
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if 'Killing application' in line:
                message = "Application has been successfully canceled"
                break;
            elif 'ApplicationNotFoundException' in line:
                message = 'Application not found to kill'
                break;
            elif 'has already finished' in line:
                message = "Application has already finished"
                break

        if message is not None:
            if not silent:
                __LOGGER__.info(message)
            return True
        else:
            __LOGGER__.error("Cannot cancel Yarn application. Here is the output from Yarn commander:")
            for l in lines:
                __LOGGER__.info(l)
            return False

class LocalAsynchronousEnvironment(ExecutionEnvironment):
    """
    An ExecutionEnvironment for local jobs that should be run asynchronously.
    """

    @staticmethod
    def run_job(job):
        """
        Runs a job asynchronously in a background process.

        1. Create a /tmp directory for this execution
        2. Serialize the Job to disk so can be read by other process
        3. Start additional process
        4. Return LocalAsynchronousJob object to caller
        """

        # Process launch with a pickled Job as file path and session.location
        driver_file_path = _os.path.join(_os.path.dirname(__file__), '_graphlabJob.py')
        path = job._get_exec_dir()
        job_path = _os.path.join(path, 'job-definition')
        _os.makedirs(path)

        ExecutionEnvironment.prepare_job_exec_dir(job, path)

        env = _gl.sys_util.make_unity_server_env()
        log_file_path = _os.path.join(path, 'execution.log')
        log_file = open(log_file_path, 'w')

        import sys
        python_exec = sys.executable
        arglist = [python_exec, driver_file_path, job_path]

        # Launch the other process
        __LOGGER__.debug("Launching process with arglist: %s" % arglist)

        if _sys.platform == 'win32':
            proc = _subprocess.Popen(arglist, env=env, stdin=_subprocess.PIPE,
                       stdout=log_file, stderr=_subprocess.STDOUT, bufsize=-1)
        else:
            proc = _subprocess.Popen(arglist, env=env, stdin=_subprocess.PIPE,
                       stdout=log_file, stderr=_subprocess.STDOUT, bufsize=-1,
                       preexec_fn=lambda: _os.setpgrp())
        __LOGGER__.debug("Process launched with pid: %d" % proc.pid)

        ret_job = _job.LocalAsynchronousJob(proc.pid,job)
        return ret_job
