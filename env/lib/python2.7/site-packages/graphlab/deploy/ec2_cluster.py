from logging import getLogger as _getLogger

from graphlab.deploy._artifact import load_artifact as _load_artifact
from graphlab.deploy._executionenvironment import Ec2ExecutionEnvironment as _ec2_execution
from graphlab.deploy.environment import _Environment
from . import _internal_utils

from graphlab.deploy import _default_session

_log = _getLogger(__name__)


def create(name, s3_path, ec2_config, num_hosts=1, additional_packages=[],
           idle_shutdown_timeout = 30 * 60):
    '''
    Create a Turi Distributed cluster in EC2.

    This will cause a cluster to start up, which will take several minutes.

    Parameters
    ----------
    name : str
        The name to use for the cluster.

    s3_path : str
        Used for storing cluster state data, such as log files. Once the cluster is created, this
        path can be used from another machine to load a handle to this cluster.

    ec2_config : Ec2Config
        Configuration information to use for starting up EC2 host(s).

    num_hosts : int, optional
        The number of EC2 host(s) to use for this environment.

    additional_packages: list[str]
        Additional packages that need to installed. The versions of these packages can be specified
        in either the pip or Anaconda format.

    idle_shutdown_timeout : int, optional
        The length of time, in seconds, the cluster is allowed to be idle before
        it will be shutdown. Default value is 30 minutes. Set to a negative value,
        for no timeout (cluster will remain running till explicitly stopped).

    Returns
    -------
    out : :py:class:`~graphlab.deploy.Ec2Cluster`
        The EC2 Cluster Handle that can be used to access and manage the Cluster.

    See Also
    --------
    graphlab.deploy.Ec2Config, graphlab.deploy.ec2_cluster.load

    Examples
    --------
    .. sourcecode:: python

        # To create the default (a single m3.xlarge host in Oregon):
        >>> my_config = graphlab.deploy.Ec2Config()
        >>> my_cluster = graphlab.deploy.ec2_cluster.create('My Test Cluster',
                                                             's3://my_bucket_name/',
                                                              my_config)

        # For a cluster of 4 "beefy" compute optimized hosts, located in North Virginia:
        >>> my_config = graphlab.deploy.Ec2Config(
                                                   instance_type = 'c4.4xlarge',
                                                   region = 'us-east-1')
        >>> my_cluster = graphlab.deploy.ec2_cluster.create('Compute Cluster',
                                                             's3://my_bucket_name/',
                                                             my_config,
                                                             num_hosts = 4)

    '''
    if not isinstance(num_hosts, int) or num_hosts <= 0:
        raise TypeError('num_hosts must be an integer, that is greater than zero.')

    if not isinstance(additional_packages, list) or any([not isinstance(x, str) for x in additional_packages]):
        raise TypeError('additional_packages must be a list of strings')
    if not isinstance(idle_shutdown_timeout, int):
        raise TypeError('\'idle_shutdown_timeout\' must be an int')

    cluster = Ec2Cluster(ec2_config, s3_path, name, num_hosts, additional_packages, idle_shutdown_timeout)

    cluster.start()
    return cluster


def load(s3_path):
    '''
    Loads an EC2 Cluster object from the s3_path.

    This will have the effect of instantiating an object locally, which contains
    all the relevant metadata associated the EC2 Cluster. This is useful for
    getting a handle of an Turi EC2 Cluster created from another machine, allowing
    you to submit jobs to it.

    Parameters
    ----------
    s3_path : str
        Used for storing cluster state data, such as log files. This path can be used from another
        machine to load a handle to this cluster.

    Returns
    -------
    out : :py:class:`~graphlab.deploy.Ec2Cluster`
        The EC2 Cluster Handle that can be used to access and manage the Cluster.

    See Also
    --------
    graphlab.deploy.ec2_cluster.create
    '''
    result = _load_artifact(s3_path + '/state')
    _default_session.save(result)
    return result


class Ec2Cluster(_Environment):
    '''
    An object for managing Turi EC2 Clusters.

    :py:func:`graphlab.deploy.ec2_cluster.create` method provides the ability to create a Ec2Cluster.
    '''
    @classmethod
    def _load_version(cls, unpickler, version):
        obj = unpickler.load()
        new = cls(obj.ec2_config, obj.s3_state_path, obj.name, obj.num_hosts, obj.additional_packages,
                  obj.idle_shutdown_timeout)
        lst = ['cluster_controller']
        _internal_utils.copy_attributes(new, obj, lst)
        return new


    def __init__(self, ec2_config, s3_state_path, name, num_hosts, additional_packages,
                 idle_shutdown_timeout):
       super(Ec2Cluster, self).__init__(name, session_aware=False)

       self.ec2_config = ec2_config
       self.s3_state_path = s3_state_path
       self.name = name
       self.num_hosts = num_hosts
       self.additional_packages = additional_packages
       self.idle_shutdown_timeout = idle_shutdown_timeout

       self.cluster_controller = None


    def __str__(self):
        result =  "S3 State Path: %s\n" % self.s3_state_path
        result += "EC2 Config   : %s\n" % self.ec2_config
        result += "Num Hosts    : %s\n" % self.num_hosts
        result += "Status       : %s" % 'Running' if self.is_running(_silent=True) else 'Stopped'
        return result

    def _get_version(self):
        return 1

    def _save(self):
        '''
        save to both local session and S3.
        '''
        self._save_to_file(self.s3_state_path + '/state')
        _default_session.save(self)

    def get_credentials(self):
        '''
        Get AWS credentials
        '''
        return self.ec2_config.get_credentials()

    def get_num_workers(self):
        '''
        Return number of workers that can be used to process jobs in parallel
        '''
        return self.num_hosts

    def is_running(self, _silent=False):
        '''
        Determine wheather or not the cluster is currently running.

        Returns
        -------
        out : bool
            Weather the cluster is currently running

        See Also
        --------
        start, stop
        '''
        # See if the local cluster_controller handle is still good.
        if self.cluster_controller is not None:
            if _ec2_execution._is_host_pingable(self.cluster_controller):
                return True

        # Check S3 for possibly updated information.
        try:
            s3_copy = load(self.s3_state_path)
        except Exception as e:
            if not _silent:
                _log.warning('Exception trying to load updated configuration '
                             'from S3 path %s. Exception: %s' % (self.s3_state_path, e))
            return False

        # Reconcile S3 info with our local info.
        if s3_copy.cluster_controller is None:
            self.cluster_controller = None
            return False
        elif self.cluster_controller == s3_copy.cluster_controller:
            return False

        # Use the update info from S3. Now check that.
        if not _silent:
            _log.info('Using updated state from S3.')
        self = s3_copy
        if _ec2_execution._is_host_pingable(self.cluster_controller):
            return True
        else:
            return False


    def start(self):
        '''
        Starts the cluster. This may take a few minutes.

        The cluster will remain running until shutdown is called.
        '''
        if self.is_running(_silent=True):
            raise RuntimeWarning('This environment has already been started')

        self.cluster_controller = _ec2_execution._start_commander_host(self.name,
                                                                       self.ec2_config,
                                                                       self.s3_state_path,
                                                                       self.num_hosts,
                                                                       self.additional_packages,
                                                                       self.idle_shutdown_timeout
                                                                       )
        self._save()

    def stop(self):
        '''
        Shuts down the cluster.

        This cluster object can be restarted, by calling `start`.
        '''
        if not self.is_running(_silent=True):
            raise RuntimeWarning('Can not stop cluster, this cluster is not running.')
        _ec2_execution._stop_cluster(self.cluster_controller)
        self.cluster_controller = None
        self._save()

    def packages(self):
        '''
        Returns a list of all installed packages in this cluster, as a list of strings that are
        formatted according to the package friendly name (name + version).

        This list includes the base packages that were installed as part of the Turi deployment,
        Anaconda packages and any user specified `additional_packages`.
        '''
        if not self.is_running(_silent=True):
            raise RuntimeWarning('The cluster must be running in order to get the list of packages.')
        return _ec2_execution._get_package_list(self.cluster_controller)


    def _submit_job(self, job_working_dir, num_workers, silent = False):
        '''This is used only by DML'''
        return _ec2_execution.submit_job(
            self, job_working_dir,
            max_concurrent_tasks = num_workers,
            silent = silent)

    def _create_job_home_dir(self, job_name):
        return _ec2_execution.create_job_home_dir(self, job_name)

    def _prepare_job_files(self, job):
        _ec2_execution.prepare_job_files(self, job)

    def _cancel_job(self, app_id, silent = False):
        '''
        Cancel a job with a given app_id
        '''
        return _ec2_execution.cancel_job(self, app_id, silent = silent)

    def _get_job_state(self, app_id, silent = False):
        '''
        Wait for a given application to enter running state

        '''
        yarn_app_states = _ec2_execution.get_job_state(self, app_id)

        if not yarn_app_states:
            raise RuntimeError('Cannot get application status from cluster.')

        return yarn_app_states
