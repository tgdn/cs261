import sys
import os
import os.path

from ConfigParser import ConfigParser, NoOptionError

import logging
_logger = logging.getLogger(__name__)

# Maps parameters to environment variable names
env_var_map = {
    'manager_endpoint': 'PS_MANAGER_ENDPOINT',
    'query_endpoint': 'PS_QUERY_ENDPOINT',
    'admin_key': 'PS_ADMIN_KEY',
    'api_key': 'PS_API_KEY',
    'verify_certificate': 'PS_VERIFY_CERTIFICATE',
    'query_timeout': 'PS_QUERY_TIMEOUT',
}

def _getter(section, option, type):
    def getter(self):
        try:
            if type == 'str':
                return self._cp.get(section, option)
            elif type == 'int':
                return self._cp.getint(section, option)
            elif type == 'float':
                return self._cp.getfloat(section, option)
            elif type == 'bool':
                return self._cp.getboolean(section, option)
            else:
                raise ValueError("Unrecognized type: %s" % (type,))
        except NoOptionError:
            return None
    return getter

def _setter(section, option):
    def setter(self, value):
        return self._cp.set(section, option, value)
    return setter

def _option_property(section, option, type, doc):
    if type not in ('str', 'int', 'float', 'bool'):
        raise ValueError("Unexpected 'type': %r" % type)
    return property(
        _getter(section, option, type),
        _setter(section, option),
        doc=doc)

class Config(object):
    """A wrapper around a ConfigParser that knows the parameters and types it
    expects."""
    _section = 'Predictive Service'

    manager_endpoint = _option_property(_section, 'manager_endpoint', 'str',
        "The manager endpoint URL")
    query_endpoint = _option_property(_section, 'query_endpoint', 'str',
        "The manager endpoint URL")

    verify_certificate = _option_property(_section,
        'verify_certificate','bool',
        "The manager endpoint URL")

    api_key = _option_property(_section, 'api_key', 'str',
        "The API key, which only has access to query and feedback and limited"
        " operations on the manager.")

    admin_key = _option_property(_section, 'admin_key', 'str',
        "The admin key, which has full access to everything.")

    query_timeout = _option_property(_section, 'query_timeout', 'float',
        "The timeout for query operations in seconds.")

    def __init__(self, cp=None):
        if cp:
            self._cp = cp
        else:
            self._cp = ConfigParser()
            self._cp.add_section(self._section)


    @classmethod
    def load(cls, name='psclient', config_file=None, raise_if_missing=True):
        """Given a config_file, loads the file and returns a config parser. It
        will also look in config_file/psclient.conf.

        config_file can also be an open file handle.

        If config_file is not specified, then the following file locations are
        searched.

            * psclient.conf in the current working directory.
            * psclient.conf in the directory of the script.
            * $XDG_CONFIG_HOME/psclient.conf
            * $XDG_CONFIG_DIRS/psclient.conf
            * ~/.psclient.conf
        """
        config_filename = name+'.conf'
        def path_gen():
            """Yields up potential paths for the config file."""
            if config_file is not None:
                # If this is a file, then we're done.
                if hasattr(config_file, 'read'):
                    yield config_file
                    return

                # Otherwise, we need to look at both config_file and
                # config_file/psclient.conf
                else:
                    yield config_file
                    yield os.path.join(config_file, config_filename)
                    return

            yield os.path.join(os.getcwd(), config_filename)
            yield os.path.join(sys.path[0], config_filename)

            try:
                home = os.path.expanduser('~')
            except:
                raise ValueError("$HOME is not set.")

            yield os.path.join(
                os.environ.get(
                    'XDG_CONFIG_HOME',
                    os.path.join(home, '.local', 'share')),
                config_filename)

            yield os.path.join(
                os.environ.get(
                    'XDG_CONFIG_DIRS',
                    os.path.join(os.sep, 'etc', 'xdg')),
                config_filename)

            yield os.path.join(
                home,
                '.'+config_filename)

        cp = ConfigParser()

        searched = []
        for path in path_gen():
            _logger.info("Trying %r", path)
            if hasattr(path, 'read'):
                _logger.info("Loading configuration from file object")
                cp.readfp(path)
                break

            if os.path.isfile(path):
                _logger.info("Loading configuration from %r", path)
                cp.read(path)
                break

            searched.append(path)
        else:
            if raise_if_missing:
                raise ValueError("Cannot find config file. Tried {}".format(
                    ", ".join([repr(s) for s in searched])))
            else:
                return None

        return cls(cp)


    def write(self, fileobject):
        self._cp.write(fileobject)

def generate_config(
        config_file,
        predictive_service_client=None,
        **kwargs
    ):
    """Given a configuration, generate a config file at config_file.

    Recommended values for config_file:

        * psclient.conf (in current working directory)
        * psclient.conf (in sys.argv[0])
        * $XDG_HOME_DIR/psclient.conf. ($XDG_HOME_DIR is ~/.local/share if
          missing)

    See psclient.connect() for where it will look for config files by default.

    Parameters
    ----------

    config_file : str or file

        The file the config will be written to.

    predictive_service_client : PredictiveService, optional

        This will be used to fill in missing values.

    **kwargs : dict

        The keys of the dict are configuration parameters.

        * If the key is missing, it will be loaded from the
        predictive_service_client, if any. If missing, it will not be added.

        * If the value are specified, these will be used, overriding whatever
        is in predictive_service_client.

        * If the value is None, then the key will not be used.

        The keys are:

        * manager_endpoint
        * query_endpoint
        * admin_key
        * api_key
        * verify_certificate
        * query_timeout

        Please refer to psclient.connect() for the meaning of these
        parameters.

    """
    cfg = Config()

    for k in (
        'manager_endpoint',
        'query_endpoint',
        'verify_certificate',
        'admin_key',
        'api_key',
        'query_timeout',
    ):
        v = None
        if k not in kwargs:
            if predictive_service_client:
                v = getattr(predictive_service_client, '_'+k)
        else:
            v = kwargs[k]

        if v is not None:
            setattr(cfg, k, v)

    if hasattr(config_file, 'write'):
        cfg.write(config_file)
    else:
        with open(os.path.expanduser(config_file), 'w') as fileobject:
            cfg.write(fileobject)

def _hide_key(param, value):
    """Used to hide keys in the logging."""
    if not param.endswith('_key'):
        return value

    if len(value) > 16:
        return value[:3]+'...'+value[-3:]
    else:
        return '********'


def load_config(params, name=None, config_file=None):
    """Loads configuration from parameters, environment variables, and a
    config file. This supports the connect() methods.

    Special note: If the params include 'manager_endpoint' and
    'query_endpoint', then it will not return query endpoints found after the
    level that the manager endpoint was found. IE, if you specify
    manager_endpoint directly but query_endpoint in a config file, it will not
    load the query endpoint from the file and instead return None.

    Parameters
    ----------

    params : dict

        A dict where the keys are the configuration parameters you want to get
        back and the values are the specified values or None.

    name : str or None

        The name specified, or None

    config_file : str or None

        The config_file specified, or None.

    Returns
    -------

    dict

        This is the params dict modified with the values set if they were
        found.
    """

    # This is the value of query_endpoint if manager_endpoint is found
    qe_sentinel = object()

    def insert_qe_sentinel():
        if 'manager_endpoint' in params and 'query_endpoint' in params \
                and params['manager_endpoint'] is not None \
                and params['query_endpoint'] is None:
            params['query_endpoint'] = qe_sentinel

    # Load the name if it is not specified.
    if name is not None:
        _logger.debug("name was specified directly as %r", name)
    else:
        try:
            name = os.environ['PS_NAME']
            _logger.info("Using $PS_NAME=%r for name", name)

        except KeyError:
            name = 'psclient'
            _logger.info("Defaulting name to %r", name)

    for k,v in params.items():
        if v is not None:
            _logger.debug("%s was directly specified as %r",
                k, _hide_key(k,v))

    insert_qe_sentinel()

    def missing():
        """Returns a list of the missing variables we need."""
        return [k
            for k,v in params.items()
            if v is None]

    def load_from_config(config):
        if config is None: return
        for k, v in params.items():
            if v is None and getattr(config, k) is not None:
                params[k] = getattr(config, k)
                _logger.debug("%s set to %r from config file",
                    k, _hide_key(k, params[k]))


    if config_file is None:
        try:
            config_file = os.environ['PS_CONFIG']
            _logger.debug("Using $PS_CONFIG=%r for config_file", config_file)
        except KeyError:
            _logger.debug("config_file and $PS_CONFIG not specified")

        insert_qe_sentinel()


    if missing() and config_file:
        _logger.debug("config_file was specified as %r. Loading from that.",
            config_file)

        load_from_config(Config.load(name=name, config_file=config_file))

        insert_qe_sentinel()


    if missing():
        _logger.debug("Examining environment variables")

        for k, v in params.items():
            if v is None:
                env_var = env_var_map[k]
                try:
                    params[k] = os.environ[env_var]
                    _logger.debug("Using $%s=%r for %s",
                        env_var, _hide_key(k, params[k]), k)
                except KeyError:
                    _logger.debug("$%s not specified.", env_var)

        insert_qe_sentinel()

    if not config_file and missing():

        load_from_config(Config.load(name=name, raise_if_missing=False))

        insert_qe_sentinel()

    # Replace the sentinel with None for query_endpoint
    if 'query_endpoint' in params and params['query_endpoint'] is qe_sentinel:
        params['query_endpoint'] = None

    # TODO: The requests module allows the verify parameter to be a string which
    # would be the directory containing the allowed SSL certificate. We should
    # probably support this.
    if 'verify_certificate' in params:
        if isinstance(params['verify_certificate'], (unicode, str)):
            params['verify_certificate'] = (
                params['verify_certificate'].lower() in (
                    'y', 'yes', 'true', 't', '1', 'verify', 'enable', 'enabled',
                ))
        elif params['verify_certificate'] is None:
            # Defaults to True
            params['verify_certificate'] = True
        else:
            params['verify_certificate'] = bool(params['verify_certificate'])

    if 'query_timeout' in params and params['query_timeout'] is not None:
        params['query_timeout'] = float(params['query_timeout'])

    return params
