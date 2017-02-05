import tempfile
import logging
import copy
import os
import sys
import uuid

from graphlab.deploy.hadoop_cluster import HadoopCluster
from graphlab.deploy.environment import LocalAsync
from graphlab.util import _make_internal_url
from graphlab.util import file_util
from . import _distributed_shell

logger = logging.getLogger(__name__)

DD_BINS_PATH = 'turi/bins/pipeline/dml/'
_script_dir = os.path.dirname(os.path.realpath(__file__))
LOCAL_DD_BINS_PATH = os.path.join(_script_dir, '..', 'deploy', 'turi_distributed', 'pipeline', 'dml')


def run(function_name, model_name, options, deploy_environment, working_dir=None):
    from graphlab.extensions import init_dml_class_registry
    from graphlab.extensions import dml_function_invocation

    if type(deploy_environment) is not HadoopCluster and \
            type(deploy_environment) is not LocalAsync:
        raise TypeError('Deployment environment %s is not supported' % str(type(deploy_environment)))

    # Working directory and num workers
    jobname = 'dml_job_%s' % str(uuid.uuid4())
    if not working_dir:
        working_dir = _dml_create_working_dir(jobname, deploy_environment)
    logger.info('Working directory: %s' % working_dir)

    num_workers = deploy_environment.get_num_workers()
    logger.info('Running with %d workers' % num_workers)

    # Substitute working_dir specific default options
    if 'model_checkpoint_path' in options and options['model_checkpoint_path'] == 'auto':
        options['model_checkpoint_path'] = _make_internal_url(os.path.join(working_dir, 'checkpoints'))

    # Serialize arguments
    init_dml_class_registry()
    args = dml_function_invocation()
    _dml_serialize_args(options, working_dir, args)

    # Make argument list for dml_commander and dml_worker
    proc_arg_list = []
    proc_arg_list.append(_get_commander_args(function_name, args.to_str(),
                                             working_dir, num_workers))
    proc_arg_list.append(_get_worker_args("${MY_RANK}", working_dir))

    # Distributed shell exec
    dshell = None
    job_handle = None
    if type(deploy_environment) is HadoopCluster:
        hadoop_cluster = deploy_environment
        dshell = _distributed_shell.HadoopDistributedShell(hadoop_cluster)
        shell_env = _get_environment_variables(hadoop_cluster)
        shell_script_file = _generate_hadoop_shell_script(proc_arg_list, shell_env, working_dir,
                                                          hadoop_cluster.turi_dist_path)
        num_containers = num_workers + 1
        job_handle = dshell.run(function_name, shell_script_file, num_containers)
    elif type(deploy_environment) is LocalAsync:
        raise NotImplementedError()
    else:
        raise ValueError('Unsupported deploy environment')

    from ._dml_job_status import DMLJobStatus
    if 'model_checkpoint_path' in options:
        return DMLJobStatus(model_name, job_handle, working_dir,
            checkpoint_path=options['model_checkpoint_path'])
    return DMLJobStatus(model_name, job_handle, working_dir)


def _generate_local_shell_script(arg_list, shell_env, working_dir, deploy_environment, **kwargs):
    """
    Generate a shell script for local distributed shell.

    Parameters
    ----------
    arg_list: list
        arg_list[0] is the commandline arguments dml_commander
        arg_list[1:] are the commandline arguments dml_worker
    shell_env: dict
        environment variables for dml_commander and dml_worker
    working_dir: str
        working directory for the job
    deploy_environment: LocalAsync
        local async environment
    """
    script_file = tempfile.NamedTemporaryFile(delete=False)
    logger.debug("script file name: " + script_file.name)

    #TODO: Make this less hard-coded
    commander_bin_path = os.path.join(LOCAL_DD_BINS_PATH, 'dml_commander_startup')
    worker_bin_path = os.path.join(LOCAL_DD_BINS_PATH, 'dml_worker_startup')

    for k, v in shell_env.items():
        script_file.write("export %s=%s\n" % (str(k), str(v)))

    script_file.write("env\n")
    script_file.write("if [ $MY_RANK -eq 0 ]; then\n")

    script_file.write("  %s " % commander_bin_path)
    for arg in arg_list[0]:
        if len(arg) > 7 and arg[0:7] == "--args=":
            script_file.write(arg[0:7] + '"' + arg[7:] + '" ')
        else:
            script_file.write(arg + " ")
    script_file.write("1> %s/commander.log.stdout " % working_dir)
    script_file.write("2> %s/commander.log.stderr " % working_dir)
    script_file.write("&\n")
    script_file.write("fi\n")

    script_file.write("%s " % worker_bin_path)
    for arg in arg_list[1]:
        script_file.write(arg + " ")
    script_file.write("1> %s/worker_${MY_RANK}.log.stdout " % working_dir)
    script_file.write("2> %s/worker.${MY_RANK}.log.stderr" % working_dir)
    script_file.write("\n")
    script_file.close()
    return script_file.name


def _generate_hadoop_shell_script(arg_list, shell_env, working_dir, turi_dist_path, **kwargs):
    """
    Generate a shell script for hadoop distributed shell.
    The shell script executes the following steps:
    1. copy dml binaries to local nodes
    2. change permission of the runnables and setup environment variables
    3. execute commander or worker code with the arg_list
    4. upload local logs to working directory

    Parameters
    ----------
    arg_list: list
        arg_list[0] is the commandline arguments dml_commander
        arg_list[1:] are the commandline arguments dml_worker
    shell_env: dict
        environment variables for dml_commander and dml_worker
    working_dir: str
        working directory for the job
    turi_dist_path: str
        path to turi distributed installation
    """
    script_file = tempfile.NamedTemporaryFile(delete=False)
    logger.debug("script file name: " + script_file.name)

    filenames_needed = ['dml_commander_startup',
                        'dml_worker_startup',
                        'libdml_toolkits.so',
                        'libdml_shared.so',
                        'libhdfs.so',
                        'libminipsutil.so',
                        'libc++abi.so.1']

    copy_cmd = "hadoop fs -copyToLocal " + turi_dist_path + "/"
    for i in filenames_needed:
        script_file.write(copy_cmd + DD_BINS_PATH + i + '\n')

    script_file.write("chmod 755 ./dml_commander_startup\n")
    script_file.write("chmod 755 ./dml_worker_startup\n")
    script_file.write("export LD_LIBRARY_PATH=${JAVA_HOME}/jre/lib/amd64/server:${LD_LIBRARY_PATH}\n")
    script_file.write("export CLASSPATH=$(hadoop classpath --glob)\n")
    for k, v in shell_env.items():
        script_file.write("export %s=%s\n" % (str(k), str(v)))

    script_file.write("env\n")
    #script_file.write("if [ $MY_RANK -eq 0 ]; then\n")
    #script_file.write("  stress --vm-bytes 4g --vm-keep -m 1 --timeout 30\n")
    #script_file.write("fi\n")
    script_file.write("if [ $MY_RANK -eq 0 ]; then\n")
    script_file.write("  echo Starting commander\n")
    script_file.write("  ./dml_commander_startup ")
    for arg in arg_list[0]:
        if len(arg) > 7 and arg[0:7] == "--args=":
            script_file.write(arg[0:7] + '"' + arg[7:] + '" ')
        else:
            script_file.write(arg + " ")
    script_file.write("> >(tee commander.log.stdout) 2> >(tee commander.log.stderr >&2)")
    script_file.write("\n")
    script_file.write("  echo Uploading commander log\n")
    script_file.write("  hadoop fs -put " + "./commander.log.stdout " +
        "/".join([working_dir, 'commander.log'])+".stdout\n")
    script_file.write("  hadoop fs -put " + "./commander.log.stderr " +
        "/".join([working_dir, 'commander.log'])+".stderr\n")
    script_file.write("else\n")
    script_file.write("  let MY_RANK=$MY_RANK-1\n")
    script_file.write("  echo Starting worker $MY_RANK\n")
    script_file.write("  ./dml_worker_startup ")
    for arg in arg_list[1]:
        script_file.write(arg + " ")
    script_file.write("> >(tee worker.log.stdout) 2> >(tee worker.log.stderr >&2)")
    script_file.write("\n")
    script_file.write("  echo Uploading worker $MY_RANK log\n")
    script_file.write("  hadoop fs -put " + "./worker.log.stdout " +
        "/".join([working_dir, "worker_${MY_RANK}.log"])+".stdout\n")
    script_file.write("  hadoop fs -put " + "./worker.log.stderr " +
        "/".join([working_dir, "worker_${MY_RANK}.log"])+".stderr\n")
    script_file.write("fi\n")
    script_file.close()
    return script_file.name


def _sanitize_internal_s3_url(input_str):
    from graphlab.connect.aws import get_credentials
    sanitized_str = input_str
    aws_id, aws_key = get_credentials()
    sanitized_str = sanitized_str.replace(aws_id, '')
    sanitized_str = sanitized_str.replace(aws_key, '')
    return sanitized_str


def _dml_serialize_args(data, working_dir, args):
    logger.info('Serializing arguments to %s' % working_dir)
    data_copy = copy.copy(data)
    internal_working_dir = _make_internal_url(working_dir)
    data_copy['__base_path__'] = internal_working_dir
    args.from_dict(data_copy, internal_working_dir)
    logger.debug('Serialized arguments: %s' % args.to_str())


def _dml_create_working_dir(jobname, deploy_environment):
    working_dir = None
    if type(deploy_environment) is HadoopCluster:
        hadoop_cluster = deploy_environment
        working_dir = hadoop_cluster._create_job_home_dir(jobname)
        # Strangely, the above _create_job_home_dir only creates
        # directory one level up from the working_dir.
        # We have to explicitly create the working directory
        # Note: only by calling hdfs_mkdir, the working dir will be created with a+rw rights :(
        file_util.hdfs_mkdir(working_dir,
                             hadoop_cluster.hadoop_conf_dir)
    elif type(deploy_environment) is LocalAsync:
        raise NotImplementedError()
    else:
        raise ValueError('Unsupported deploy environment')
    logger.debug('Working directory created: %s' % working_dir)
    return working_dir


def _get_commander_args(function_name, data,
                        working_dir,
                        num_workers,
                        shared_lib='./libdml_toolkits.so',
                        cluster_type='standalone_passive',
                        output_name='out',
                        **kwargs):
    """
    Get a list of arguments for dml_commander_startup
    """
    args = dict()
    # from arguments
    args['function'] = function_name
    args['args'] = data
    args['num_nodes'] = num_workers
    args['working_dir'] = _make_internal_url(working_dir)

    # from optional arguments
    args['shared_lib'] = shared_lib
    args['cluster_type'] = cluster_type
    args['output_name'] = output_name

    # from kwargs, could overwrite existing args
    accepted_args = list(args.keys()) + ['check_hdfs', 'startup_timeout',
                                         'metric_server_address_file',
                                         'metric_server_port']
    for key in accepted_args:
        if key in kwargs:
            args[key] = kwargs[key]

    # return a formated list
    return ['--%s=%s' % (k, v) for k, v in args.items()]


def _get_worker_args(worker_id, working_dir, **kwargs):
    """
    Get a list of arguments for dml_worker_startup
    """
    args = dict()
    args['worker_id'] = worker_id
    args['working_dir'] = _make_internal_url(working_dir)

    accepted_args = ['check_hdfs', 'startup_timeout', 'consensus_address'] + list(args.keys())
    for key in accepted_args:
        if key in kwargs:
            args[key] = kwargs[key]
    return ['--%s=%s' % (k, v) for k, v in args.items()]


def _get_environment_variables(cluster):
    """
    Returns the environment variables for running dml_commander and dml_worker.
    """
    # Make environment variables for the shell commands
    shell_env = {}
    shell_env["OMP_NUM_THREADS"] = cluster.num_vcores
    shell_env["GRAPHLAB_MEMORY_LIMIT_IN_MB"] = cluster.container_size * .8
    shell_env["TURI_PORT_START"] = cluster.start_port
    shell_env["TURI_PORT_END"] = cluster.end_port
    # Overwrite environment for SFrame cache capacity. Being conservative, we use default 1/5 of the container size.
    for key in ['GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY', 'GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE']:
        if key in os.environ:
            shell_env[key] = os.environ[key]
        else:
            shell_env[key] = cluster.container_size * 1024 * 1024 * .2

    # Overwrite environment for SFrame cache file location.
    if cluster.node_tmp_dir:
        shell_env['GRAPHLAB_CACHE_FILE_LOCATIONS'] = cluster.node_tmp_dir
    if cluster.hdfs_tmp_dir:
        shell_env['GRAPHLAB_CACHE_FILE_HDFS_LOCATION'] = cluster.hdfs_tmp_dir
    return shell_env


def _dml_read_app_progress_file(working_dir):
    progress_file = os.path.join(working_dir, 'progress.log')
    lines = file_util.read(progress_file)
    return lines


def _dml_read_app_metric_server(working_dir):
    if sys.version_info.major == 2:
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen

    metric_server_address = os.path.join(working_dir, 'metric_server_address')
    url = file_util.read(metric_server_address)
    if not url or url[-1] != '$':
        return ''
    else:
        url = url[:-1] + '/progress'
        logger.info('Open url %s' % url)
        return urlopen(url).read()
