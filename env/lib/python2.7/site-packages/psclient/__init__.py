"""
psclient is a Python library to interact with Turi Predictive Services.

It provides a convenient interface for all management and query APIs exposed by a
Predictive Service deployment. Additionally it allows to create a cloud-based
deployment from scratch.
"""

__version__ = '2.0'

from .constants import _DEFAULT_ADMIN_PORT, _DEFAULT_ADMIN_UI_PORT
from ._ec2_config import Ec2Config
from .policy import EpsilonGreedyPolicy, ProbabilityPolicy

import logging as _logging

# Disable the unconfigured logging error message.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Swallow all but CRITICAL log messages in boto
_logging.getLogger('boto').setLevel(_logging.CRITICAL)

# Certain modules will, by default, produce log messages on stderr. This is
# the handler used to manage that.
_default_log_handler = _logging.StreamHandler()
_default_log_handler.setFormatter(
    _logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
_default_log_handler.setLevel(_logging.INFO)

# These are the modules under psclient that will be verbose.
_verbose_modules = (
    'create_fn',
    'upgrade_fn',
    'cluster_client',
    'aws',
)

def enable_default_logging():
    """Enables the default logging to stderr.

    Note that this is enabled by default.
    """
    for module in _verbose_modules:
        logger = _logging.getLogger(__name__+'.'+module)
        logger.setLevel(_logging.INFO)
        logger.addHandler(_default_log_handler)

# enabled by default
enable_default_logging()

def disable_default_logging():
    """Disables the default logging to stderr.

    By default, some of the modules in psclient log their messages to stderr.
    If this is undesirable, you can call this function to disable that kind of
    logging.
    """
    for module in _verbose_modules:
        _logging.getLogger(__name__+'.'+module).removeHandler(_default_log_handler)


def connect(
        manager_endpoint=None,
        admin_key=None,
        query_endpoint=None,
        verify_certificate=None,
        query_timeout=None,
        name=None,
        config_file=None):
    """Instantiate a PredictiveService object from config files or environment
    variables.

    The client needs a manager_endpoint, query_endpoint and admin_key to be able
    to conect to a deployment. These can be specified directly, or they can be
    loaded from the environment variables or configuration files.

    The logic for determining the parameters is as follows. These steps are
    followed until all required parameters are found.

    * If the parameters are specified directly, use those.
    * If config_file or the environment variable $PS_CONFIG is specified:

        * If it's a file object, load directly from the object.
        * If it's a directory, look for <config_file>/<name>.conf.
        * If it's a file name, load from that.

    * Use the environment variables $PS_QUERY_ENDPOINT, $PS_API_KEY, or
      $PS_VERIFY_CERTIFICATE, if specified.
    * If config_file and $PS_CONFIG is not specified, then the following
      locations are searched for a configuration file:

        * <name>.conf in the current working directory.
        * <name>.conf in the directory of the script.
        * $XDG_CONFIG_HOME/<name>.conf, or ~/.local/share/<name>.conf if
          $XDG_CONFIG_HOME is not specified.
        * $XDG_CONFIG_DIRS/<name>.conf, or /etc/xdg/<name>.conf if
          $XDG_CONFIG_DIRS is not specified.
        * ~/.<name>.conf

    **Config File Format**

    The config file is standard windows or Python .ini format. It should look
    something like this:

    |    [Predictive Service]
    |    manager_endpoint = https://.../
    |    query_endpoint = https://.../
    |    admin_key = 123...abc

    Configuration files can be generated with
    PredictiveService.generate_config(). One should also be generated when
    psclient.create() was successfully executed.

    Parameters
    ----------
    manager_endpoint : str, optional
        The manager service endpoint.

    query_endpoint: str, optional
        The query service endpoint. If not specified, the manager will be
        queried for it.

    admin_key : str, optional
        The admin key. If not specified, the environment variable PS_ADMIN_KEY
        will be used. If that is not set, a found config file will be used.

    name : str, optional
        The name of the service. This is used to find the configuration file.
        It will look for files named <name>.conf. By default, this is
        $PS_NAME, or if that is unspecified, 'psclient'.

    config_file : str or file, optional
        The configuration file to use. This can be a file object or it can be
        a path to a directory or a file. If it is a directory, then it looks
        for <name>.conf in that directory. If it is not specified, and the
        configuration parameters are not specified or found in the environment
        variables, then a series of locations will be searched for the
        appropriate configuration file.

    Returns
    -------
    PredictiveService
        A handle to the deployment.
    """
    from .connect_fn import connect
    return connect(
        manager_endpoint=manager_endpoint,
        admin_key=admin_key,
        query_endpoint=query_endpoint,
        verify_certificate=verify_certificate,
        query_timeout=query_timeout,
        name=name,
        config_file=config_file)

def create(name, ec2_config, state_path, license_file_path=None, num_hosts=1,
           description=None, api_key=None, admin_key=None,
           ssl_credentials=None, cors_origin='',
           port=_DEFAULT_ADMIN_PORT, admin_ui_port=_DEFAULT_ADMIN_UI_PORT,
           scheme='internet-facing',
           config_file=None):
    '''
    Launch a Predictive Services deployment. This deployment can currently be
    launched on EC2 by specifying an EC2 environment.

    Parameters
    ----------
    name : str
        The name of the deployment to be launched. This name can only contain:
        a-z, A-Z and hyphens.

    ec2_config : :class:`~psclient.Ec2Config`
        Must be an EC2 Configuration object used for starting up EC2 host(s).
        This configuration should contain the instance type, region, aws
        credentials, and other information used to launch EC2 host(s).

    state_path : str
        S3 path used to store state for the deployment.

    license_file_path : str, optional
        The path to the local license file that will be used to launch the
        deployment. If not specified, the default license file path is
        ~/.graphlab/config

    num_hosts : int, optional
        The number of EC2 host(s) to use for this deployment. The default number
        of EC2 host(s) is 1.

    description : str, optional
        Description of the deployment.

    api_key : str, optional
        An API key for endpoint queries. If an API key is not specified, it will
        be auto-generated. API keys can be retrieved through the returned
        PredictiveService object.

    admin_key : str, optional
        An admin key used for control operations. If an admin key is not
        specified, it will be auto generated. Admin keys can be retrieved
        through the returned PredictiveService object.

    ssl_credentials : tuple of len three, with types: str, str, bool.
        The first string is the path to the private key file. The second string
        is the path to public key certificate. The third denotes whether the
        certificates are self signed (and any client should not verify the
        certificate).

        These files must be in the precise format AWS expects. Such a private
        key and a self-signed certificate can be generated using openssl with
        following commands:

        >>> openssl genrsa 1024 > privatekey.key
        >>> openssl req -new -key privatekey.key -out CSR.csr
        >>> openssl x509 -req -days 365 -in CSR.csr -signkey privatekey.key
                    -out certificate.crt

        If a tuple is not given, requests will be served over HTTP rather than
        HTTPS (i.e., encryption will not be used).

    cors_origin : str, optional
        The string value to use as HTTP header Access-Control-Allow-Origin,
        in order to support Cross-Origin Resource Sharing as described in
        https://developer.mozilla.org/en-US/docs/Web/HTTP/Access_control_CORS.
        The default value is ''. If '' (empty string) is specified, CORS
        support will be disabled. If the string value is '*', CORS support
        will be enabled for all URIs. If the string value is
        'https://www.example.com', CORS support will be enabled for
        'https://www.example.com' only.

    port : int, optional
        The administration port for the server, default is 8088. If you have
        a firewall enabled in your enterprise, you need to open this port in
        order to manage the deployment. The port value cannot be 9005, 9006,
        19006 as they are reserved.

    admin_ui_port : int, optional
        The administration UI (dashboard) port for the server, default is 8888.
        If you have a firewall enabled in your enterprise, you need to open this
        port in order to be able to access the dashboard. The port value cannot
        be 9005, 9006, 19006 as they are reserved.

    scheme : str, optional
        The type of a LoadBalancer. By default, Elastic Load Balancing creates an
        "internet-facing" LoadBalancer with a publicly resolvable DNS name, which
        resolves to public IP addresses.
        Specify the value "internal" for this option to create an internal LoadBalancer
        with a DNS name that resolves to private IP addresses.

    config_file : str, optional
        A file path to write a config file at if specified. A config file can be
        used for :func:`psclient.connect`. It can also be generated after the
        deployment is created, through ``generate_config``.

    Returns
    -------
    PredictiveService
        A handle to the deployment.

    See Also
    --------
    psclient.connect, psclient.predictive_service_client.PredictiveService,
    psclient.predictive_service_client.PredictiveService.generate_config

    Examples
    --------
    .. sourcecode:: python

        ec2 = psclient.Ec2Config(region='us-west-2',
                                 instance_type='m3.xlarge',
                                 aws_access_key_id='YOUR_ACCESS_KEY',
                                 aws_secret_access_key='YOUR_SECRET_KEY')

        ps = psclient.create('my-deployment',
                             ec2,
                             's3://my-bucket/some-name',
                             num_hosts=3)
    '''
    from .create_fn import create
    return create(
        name=name,
        ec2_config=ec2_config,
        state_path=state_path,
        license_file_path=license_file_path,
        num_hosts=num_hosts,
        description=description,
        api_key=api_key,
        admin_key=admin_key,
        ssl_credentials=ssl_credentials,
        cors_origin=cors_origin,
        port=port,
        scheme=scheme,
        config_file=config_file)

def get_version():
    return __version__
