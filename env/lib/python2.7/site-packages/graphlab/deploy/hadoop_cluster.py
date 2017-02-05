import os as _os
import logging as _logging
import tempfile as _tempfile

import graphlab as _gl

from .environment import _Environment
from graphlab.util import file_util as _file_util
from ._executionenvironment import HadoopExecutionEnvironment as \
    _HadoopExecutionEnvironment

import sys as _sys
if _sys.version_info.major == 2:
    import ConfigParser as _ConfigParser
else:
    import configparser as _ConfigParser


_log = _logging.getLogger(__name__)

def create(
        name,
        turi_dist_path,
        hadoop_conf_dir = None,
        num_containers = None, container_size = None, num_vcores = None,
        start_port = None, end_port = None,
        additional_packages = None):
    '''
    Create a Turi Distributed cluster in Hadoop.

    Parameters
    -----------

    name : str
        A name for newly created cluster, this name is used in your local session
        so that you may easily load it back using:

            >>> import graphlab
            >>> graphlab.deploy.environments[<cluster-name>]

    turi_dist_path : str
        The location where Turi Distributed is installed. This usually comes from
        your Hadoop Administrator. This path must be a valid HDFS path.

    hadoop_conf_dir : str, optional
        Hadoop configure directory where Hadoop configuration files are stored.
        If not given, the configuration file is automatically searched in your
        CLASSPATH. hadoop_conf_dir must be a local file path.

    num_containers : int, optional
        The number of containers to use for this environment. If not given,
        default value is decided by you Turi Distributed administrator when
        installing Turi Distributed.

    container_size : int, optional
        The memory in MB required for job execution.  If not given,
        default value is decided by you Turi Distributed administrator when
        installing Turi Distributed.

    num_vcores : int, optional
        The number of virtuals cores to use, must be at least two. If not given,
        default value is decided by you Turi Distributed administrator when
        installing Turi Distributed.

    additional_packages : list [str], optional
        Additional packages you want to use in your Hadoop Execution Environment.
        The package can be in one of the following format:

            <package-name>
            <packge-name>=<package-version>
            <package-name>==<package-version>

        You may use either Conda package or Pypi package.

        Any package you listed here must have been added to the package list by
        your Hadoop Administrator to the Turi Distributed installation. You may
        use the following command to list all available packages in the Turi
        Distributed installation:

            >>> import graphlab
            >>> hadoop_cluster.show_available_packages()


    Returns
    --------
    cluster | a HadoopCluster object
        A handle to the Hadoop execution environment for your distributed job
        execution

    Examples
    ---------

    To create a Hadoop cluster execution environment using default configuration:

        >>> import graphlab
        >>> hadoop_cluster = graphlab.deploy.hadoop_cluster.create(
        ...         name = 'my-first-cluster',
        ...         turi_dist_path = 'hdfs://namenode:port/turi-dist-path'
        )

    To create a Hadoop cluster execution environment with all explicit config:

        >>> import graphlab
        >>> hadoop_cluster = graphlab.deploy.hadoop_cluster.create(
        ...         name = 'my-first-cluster',
        ...         turi_dist_path = 'hdfs://namenode:port/turi-dist-path'
        ...         hadoop_conf_dir = '<path-to-hadoop-conf>',
        ...         num_containers = 10,
        ...         num_vcores = 8,
        ...         container_size = 8192,
        ...         additional_packages = ['nltk==3.0.3']
        )

    '''

    hadoop_conf_dir = _file_util.expand_full_path(hadoop_conf_dir) if hadoop_conf_dir else None
    _validate_turi_distr_param(turi_dist_path, hadoop_conf_dir)

    if not isinstance(name, basestring):
        raise TypeError('Cluster name has to be a string.')

    if additional_packages is not None:
        if isinstance(additional_packages, basestring) :
            additional_packages = [additional_packages]

        if not hasattr(additional_packages, '__iter__'):
            raise TypeError('"additional_packages" parameter has to be iterable.')

    # Now create a HadoopCluster object
    cluster = HadoopCluster(name, turi_dist_path, hadoop_conf_dir,
                            num_containers, container_size, num_vcores,
                            additional_packages)

    # Save to local session and overwrite if exists
    if cluster._session.exists(cluster.name, HadoopCluster._typename):
        _log.warning('Overwriting existing Hadoop Cluster "%s" in local session.' % cluster.name)
        _gl.deploy.environments.delete(cluster, silent=True)
    _gl.deploy._default_session.register(cluster)
    cluster.save()

    return cluster

def upload_packages(
    turi_dist_path, filename_or_dir,
    hadoop_conf_dir = None, force=False):
    '''
    Upload a package to the available packages for this Hadoop Turi Distributed
    installation.  Files must be a valid PyPI package.  You may download packages
    from PyPI with the commands

        >>> mkdir <directory_name>
        >>> pip install --download <directory_name> <package-name>

        then

        >>> graphlab.deploy.hadoop_cluster.upload_packages(<turi_dist_path>, <path_to_directory>)

    These packages will be available for future work on the cluster.

    Parameters
    -----------
    turi_dist_path : str
        The location where Turi Distributed is installed. This usually comes from
        your Hadoop Administrator. This path must be a valid HDFS path.

    filename_or_dir :  str
        A file or directory containing files to upload, the file(s) must be a
        correct package for your target host's operating system in your Hadoop
        setup.

    hadoop_conf_dir : str, optional
        Hadoop configure directory where Hadoop configuration files are stored.
        If not given, the configuration file is automatically searched in your
        CLASSPATH. hadoop_conf_dir must be a local file path.

    force: boolean, optional
        Boolean, whether to force overwrite if file exists

    Returns
    -------
    '''
    hadoop_conf_dir = _file_util.expand_full_path(hadoop_conf_dir) if hadoop_conf_dir else None
    _validate_turi_distr_param(turi_dist_path, hadoop_conf_dir)

    dest = turi_dist_path + HadoopCluster._DIST_USER_PKG
    if _os.path.isdir(filename_or_dir):
        for root, directories, filenames in _os.walk(filename_or_dir):
            for f in filenames:
                full= _os.path.join(root, f)
                _file_util.upload_to_hdfs(full, dest,
                    hadoop_conf_dir=hadoop_conf_dir, force=force)
    else:
        _file_util.upload_to_hdfs(filename_or_dir, dest,
            hadoop_conf_dir=hadoop_conf_dir, force=force)

def remove_package(turi_dist_path, filename, hadoop_conf_dir = None):
    '''
    Remove a package from the available packages for this Hadoop Turi Distributed
    installation. This package will no longer be available for installation.

    Parameters
    -----------
    turi_dist_path : str
        The location where Turi Distributed is installed. This usually comes from
        your Hadoop Administrator. This path must be a valid HDFS path.

    filename :  str
        File name of the package to remove from the Turi Distributed installation

    hadoop_conf_dir : str, optional
        Hadoop configure directory where Hadoop configuration files are stored.
        If not given, the configuration file is automatically searched in your
        CLASSPATH. hadoop_conf_dir must be a local file path.

    '''
    hadoop_conf_dir = _file_util.expand_full_path(hadoop_conf_dir) if hadoop_conf_dir else None
    _validate_turi_distr_param(turi_dist_path, hadoop_conf_dir)

    full = turi_dist_path + HadoopCluster._DIST_USER_PKG + "/" + filename
    _file_util.remove_hdfs(full, hadoop_conf_dir=hadoop_conf_dir)

def show_available_packages(turi_dist_path, hadoop_conf_dir = None):
    '''
    Show all availabe packages in Hadoop Turi Distributed installation

    turi_dist_path : str
        The location where Turi Distributed is installed. This usually comes from
        your Hadoop Administrator. This path must be a valid HDFS path.

    hadoop_conf_dir : str, optional
        Hadoop configure directory where Hadoop configuration files are stored.
        If not given, the configuration file is automatically searched in your
        CLASSPATH. hadoop_conf_dir must be a local file path.

    Returns
    -------
    out : dict
        Dict of two lists, default_packages in the format:

            "rsa==3.1.4",
            "scikit-learn==0.16.1",
            "scipy==0.15.1"
        and user_packages, additional PyPi packages which have been uploaded to the Turi Distributed
        installation.  user_packages has the format:

            "names-0.3.0.tar.gz",
            "boto-2.33.0-py2.py3-none-any.whl",
            ...

    '''
    hadoop_conf_dir = _file_util.expand_full_path(hadoop_conf_dir) if hadoop_conf_dir else None
    _validate_turi_distr_param(turi_dist_path, hadoop_conf_dir)

    conda_list = turi_dist_path + HadoopCluster._DIST_CONDA_LIST
    user_list = turi_dist_path + HadoopCluster._DIST_USER_PKG
    packages = _file_util.read_file_to_string_hdfs(conda_list, hadoop_conf_dir=hadoop_conf_dir)
    if packages is None:
        raise RuntimeError("It seems like you do not have a valid Turi Distributed"
        " installation. Please contact your Hadoop administrator.")

    lines = packages.split(_os.linesep)
    output_lines = []
    for l in lines:
        splited = l.split()
        if len(splited) == 3:
            output_lines.append('%s==%s' % (splited[0], splited[1]))

    result = {'default_packages': output_lines}
    user_add = _file_util.list_hdfs(user_list, hadoop_conf_dir=hadoop_conf_dir)
    user = [_os.path.basename(x['path']) for x in user_add]
    result['user_packages'] = user
    return result

def _validate_turi_distr_param(turi_dist_path, hadoop_conf_dir = None):
    if not type(turi_dist_path) in [str, unicode]:
        raise TypeError('"turi_dist_path" needs to be a string')

    if hadoop_conf_dir is not None and not _os.path.isdir(hadoop_conf_dir):
        raise ValueError('"%s" has to be a local folder.' % hadoop_conf_dir)

    if not _file_util.hdfs_test_url(turi_dist_path, \
                                    hadoop_conf_dir = hadoop_conf_dir):
        raise ValueError('Hadoop path "%s" does not exist.' % turi_dist_path)


class HadoopCluster(_Environment):
    '''
    HadoopCluster represents a Turi Distributed cluster backed up by a Hadoop
    system. It uses Hadoop's resource manager to manage job resources and submit
    Turi's job as Yarn job to Hadoop.

    HadoopCluster should not be explicitly instantiated. The cluster should only
    be created through one of the following two ways:


        :py:func:`~graphlab.deploy.hadoop_cluster.create`

        :py:obj:`~graphlab.deploy.environments` [name]

    '''
    _HADOOP_CLUSTER_VERSION = 1
    _DIST_INI = "/turi/conf/default-conf.ini"
    _DIST_CONDA_LIST = "/turi/bins/runtime/conda_list/package_list.txt"
    _DIST_USER_PKG = "/turi/lib"

    @classmethod
    def _load_version(cls, unpickler, version):
        '''
        Load HadoopCluster object from a pickled file
        '''
        obj = unpickler.load()
        new = cls(
            name = obj.name,
            turi_dist_path = obj.turi_dist_path,
            hadoop_conf_dir = obj.hadoop_conf_dir,
            num_containers = obj.num_containers,
            container_size = obj.container_size,
            num_vcores = obj.num_vcores,
            additional_packages = obj.additional_packages,
            )
        return new

    @staticmethod
    def _get_hadoop_jar():
        '''
        Get the client jar file that is used for submitting Yarn application
        '''
        return _os.path.join(_os.path.dirname(
                _os.path.realpath(__file__)), "graphlabHadoopYarn-1.0.jar")

    def __init__(self,
                name,
                turi_dist_path,
                hadoop_conf_dir = None,
                num_containers = None, container_size = None, num_vcores = None,
                additional_packages = None):

        '''
        Initializes a new HadoopCluster object

        '''
        super(HadoopCluster, self).__init__(name, session_aware = False)

        self.name = name
        self.turi_dist_path = turi_dist_path
        self.additional_packages = additional_packages
        self.hadoop_conf_dir = hadoop_conf_dir

        config = self._read_cluster_state()
        self.num_containers = num_containers if num_containers is not None else \
                              config.getint('runtime', 'num_containers')
        self.container_size = container_size if container_size is not None else \
                              config.getint('runtime', 'container_size')
        self.num_vcores = num_vcores if num_vcores is not None else \
                         config.getint('runtime', 'num_vcores')
        self.start_port = config.getint('runtime', 'start_port')
        self.end_port = config.getint('runtime', 'end_port')

        self.node_tmp_dir =  config.get('turi-distrib','node_tmp_dir')
        self.hdfs_tmp_dir =  config.get('turi-distrib','hdfs_tmp_dir')
        #internal use only
        if config.has_option('turi-distrib', 'conda_installed_dir'):
          self._conda_installed_dir = config.get('turi-distrib', 'conda_installed_dir')
        else: 
          self._conda_installed_dir = None


    def _get_version(self):
        return self._HADOOP_CLUSTER_VERSION

    def __str__(self):
        ret = 'Hadoop Cluster:\n'
        ret += '\tName:                    : %s\n' % self.name
        ret += '\tCluster path             : %s\n' % self.turi_dist_path

        if self.hadoop_conf_dir:
            ret += '\tHadoop conf dir          : %s\n' % self.hadoop_conf_dir

        ret += '\n'
        ret += '\tNumber of Containers:    : %s\n' % self.num_containers
        ret += '\tContainer Size (in mb)   : %s\n' % self.container_size
        ret += '\tContainer num of vcores  : %s\n' % self.num_vcores
        ret += '\tPort range               : %s - %s\n' % (self.start_port, self.end_port)
        ret += '\tNode temp directory      : %s\n' % self.node_tmp_dir
        ret += '\tHDFS temp directory      : %s\n' % self.hdfs_tmp_dir
        ret += '\n'
        ret += '\tAdditional packages      : %s\n' % self.additional_packages
        ret += '\n'
        return ret

    def __repr__(self):
        return self.__str__()

    def is_running(self):
        '''
        Returns whether or not the cluster is running
        '''
        return True

    def get_num_workers(self):
        '''
        Return number of workers that can be used to process jobs in parallel
        '''
        return self.num_containers

    def _submit_job(self, job_working_dir, num_workers, silent = False):
        return _HadoopExecutionEnvironment.submit_job(
            self,
            job_working_dir, silent)

    def _create_job_home_dir(self, job_name):
        return _HadoopExecutionEnvironment.create_job_home_dir(self, job_name)

    def _prepare_job_files(self, job):
        '''
        Prepare all job files and upload to HDFS
        '''
        _HadoopExecutionEnvironment.prepare_job_files(self, job)

    def _cancel_job(self, app_id, silent = False):
        '''
        Cancel a given job
        '''
        return _HadoopExecutionEnvironment.cancel_yarn_application(
            self, app_id, silent = silent)

    def _get_job_state(self, app_id, silent = False):
        '''
        Wait for a given application to enter running state

        '''
        yarn_app_states = _HadoopExecutionEnvironment.get_yarn_application_state(
            self, app_id, silent = True)

        if not yarn_app_states:
            raise RuntimeError('Cannot get application status from Yarn. '
                                'Please check if Yarn is in healthy state.')

        return yarn_app_states

    def _read_cluster_state(self):
        local_cluster_config_file = _tempfile.mktemp(prefix='hadoop-conf-')
        try:
            remote_cluster_config_file = "%s%s" % (self.turi_dist_path, HadoopCluster._DIST_INI)

            if not _file_util.hdfs_test_url(remote_cluster_config_file, \
                                            hadoop_conf_dir = self.hadoop_conf_dir):
                raise ValueError('Path "%s" does not seem like a valid Turi Distributed '
                                 'installation.' % self.turi_dist_path)

            _file_util.download_from_hdfs(
                            hdfs_path = remote_cluster_config_file,
                            local_path = local_cluster_config_file,
                            hadoop_conf_dir=self.hadoop_conf_dir)

            config = _ConfigParser.ConfigParser()
            config.read(local_cluster_config_file)
            return config

        finally:
            if _os.path.exists(local_cluster_config_file):
                _os.remove(local_cluster_config_file)
