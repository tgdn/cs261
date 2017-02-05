import os as _os
import graphlab.version_info as _version_info
from pkg_resources import parse_version as _parse_version
import logging as _logging
import sys as _sys

if _sys.version_info.major == 2:
    from urllib2 import urlopen as _urlopen
    import ConfigParser as _ConfigParser
else:
    from urllib.request import urlopen as _urlopen
    import configparser as _ConfigParser

__GLCREATE_CURRENT_VERSION_URL__ = "https://turi.com/files/glcreate_current_version"

__LOGGER__ = _logging.getLogger(__name__)

def get_newest_version(timeout=5, _url=__GLCREATE_CURRENT_VERSION_URL__):
    """
    Returns the version of GraphLab Create currently available from turi.com.
    Will raise an exception if we are unable to reach the turi.com servers.

    Parameters
    ----------
    timeout: int
        How many seconds to wait for the remote server to respond

    url: string
        The URL to go to to check the current version.

    Returns
    -------
    version : str
       The version number of the current graphlab create.
    """
    request = _urlopen(url=_url, timeout=timeout)
    version = request.read()
    if version:
        version = version.decode()
    __LOGGER__.debug("current_version read %s" % version)
    return version


def get_major_version(version_str):
    """
    Returns the major version of GraphLab Create currently running.

    Returns
    -------
    version : str
       The major version number of the current graphlab create.
    """
    return '.'.join(version_str.split('.')[:2])


def perform_version_check(configfile=(_os.path.join(_os.path.expanduser("~"), ".graphlab", "config")),
                          _version_url=__GLCREATE_CURRENT_VERSION_URL__,
                          _outputstream=_sys.stderr):
    """
    Checks if currently running version of GraphLab Create is less than the
    version available from turi.com. Prints a message if the turi.com servers
    are reachable, and the current version is out of date. Does nothing
    otherwise.

    If the configfile contains a key "skip_version_check" in the Product
    section with non-zero value, this function does nothing.

    Returns
    -------
    bool
        returns True if a message is printed, and returns False otherwise.
    """
    skip_version_check = False
    try:
        if (_os.path.isfile(configfile)):
            config = _ConfigParser.ConfigParser()
            config.read(configfile)
            section = 'Product'
            key = 'skip_version_check'
            skip_version_check = config.getboolean(section, key)
            __LOGGER__.debug("skip_version_check=%s" % str(skip_version_check))
    except:
        # eat all errors
        pass

    # skip version check set. Quit
    if not skip_version_check:
        try:
            latest_version = get_newest_version(timeout=1,
                                                _url=_version_url).strip()
            if _parse_version(latest_version) > _parse_version(_version_info.version.replace(".gpu","")):
                msg = ("A newer version of GraphLab Create (v%s) is available! "
                       "Your current version is v%s.\n"
                       "You can use pip to upgrade the graphlab-create package. "
                       "For more information see https://turi.com/products/create/upgrade.\n") % \
                      (latest_version, _version_info.version)
                _outputstream.write(msg)
                return True
        except:
            # eat all errors
            pass
    return False

