'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''
import json as _json
import logging as _logging
import os as _os
import requests as _requests

from .util import sys_info as _sys_info

import sys as _sys
if _sys.version_info.major == 2:
    import ConfigParser as _ConfigParser
else:
    import configparser as _ConfigParser


__LOGGER__ = _logging.getLogger(__name__)

__default_dato_endpoint = 'https://beta.graphlab.com'
__section = 'Product'
__key = 'product_key'
__license = 'license_info'
__default_config_path = _os.path.join(_os.path.expanduser("~"), ".graphlab", "config")


def get_product_key(file=__default_config_path):
    """
    Returns the product key found in file, which by default is ~/.graphlab/config
    or in environment variable GRAPHLAB_PRODUCT_KEY.

    Note: Environment variable takes precedence over config file.

    @param file optional parameter to specify which file to use for configuration
           (defaults to ~/.graphlab/config)

    @return Product key string, or None if not found.
    """
    PRODUCT_KEY_ENV = 'GRAPHLAB_PRODUCT_KEY'
    if PRODUCT_KEY_ENV in _os.environ:
        return _os.environ[PRODUCT_KEY_ENV]

    import graphlab.connect as _mt
    # see if in ~/.graphlab/config
    if (_os.path.isfile(file)):
        try:
            config = _ConfigParser.ConfigParser()
            config.read(file)
            product_key = config.get(__section, __key)
            if product_key == -1:
                raise BaseException() # will fall into except block below
            else:
                # Don't clobber os.environ[PRODUCT_KEY_ENV] -- that would
                # prevent future calls to get_product_key from returning
                # different values. Each call should always check the
                # config file since it can change over time.
                return str(product_key).strip('"\'')
        except:
            msg = "License check failed: Unable to validate license."
            _mt._get_metric_tracker().track('get_product_key.config_parser_error')
            raise KeyError(msg)
    else:
        return None


def set_product_key(product_key, file=__default_config_path):
    """
    Sets the product key provided in file, which by default is ~/.graphlab/config
    Overwrites any existing product key in that file.

    Note: Environment variable GRAPHLAB_PRODUCT_KEY takes precedence over the
    config file and is not affected by this function.

    Parameters
    ----------
    product_key : str
        The product key

    file : str, optional
        Specifies which file to use for configuration (defaults to ~/.graphlab/config)
    """
    import graphlab.connect as _mt
    try:
        config = _ConfigParser.ConfigParser()
        config.read(file)
        if not(config.has_section(__section)):
            config.add_section(__section)
        config.set(__section, __key, product_key)
        with open(file, 'w') as config_file:
            config.write(config_file)
        _mt._get_metric_tracker().track('set_product_key.succeeded')
    except:
        _mt._get_metric_tracker().track('set_product_key.config_parser_error')
        __LOGGER__.warn('Unable to write GraphLab Create product key to %s. Ensure that this user account \
                        has write permission to %s to save the product key locally.' % (file, file))
        _os.environ['GRAPHLAB_PRODUCT_KEY'] = str(product_key)


def _get_license_info(config_path=__default_config_path):
    """
    Gets the license info from Turi's server. If unable to reach Turi's server, use local config file
    as fall back.

    @param config_path optional parameter to specify which file to use for configuration (defaults to ~/.graphlab/config)

    @return License info string, or None if not found.
    """
    import graphlab.connect as _mt
    # check turi.com for license info first (always prefer online license check)
    product_key = get_product_key(file=config_path)
    try:
        data = {
            'event_name': 'license-check',
            'request_from': 'GraphLab-Create',
            'value': 1,
            'distinct_id': _sys_info.get_distinct_id(),
            'version': _sys_info.get_version(),
            'isgpu': _sys_info.get_isgpu(),
            'build_number': _sys_info.get_build_number(),
            'properties': _sys_info.get_sys_info(),
            'product_key': product_key
        }
        response = _requests.post(__default_dato_endpoint + '/license/check', data=_json.dumps(data),
                                  timeout=3, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            # write this in config file
            blob = response.content.decode()
            __set_license_info(product_key, blob, config_path)
            return blob
        elif response.status_code == 404:
            blob = _json.loads(response.content)
            if 'error' in blob:
                msg = blob['error']
                raise RuntimeError(msg)
    except _requests.exceptions.RequestException:
        _mt._get_metric_tracker().track('get_license_info.online_failed.')

    # 1st cache fallback, see if it is in the environment variable
    if 'GRAPHLAB_LICENSE_INFO' in _os.environ:
        return _os.environ['GRAPHLAB_LICENSE_INFO']

    # 2nd cache fallback, see if it is in ~/.graphlab/config
    if (_os.path.isfile(config_path)):
        config = _ConfigParser.ConfigParser()
        config.read(config_path)
        try:
            license_info = config.get(__section, __license)
            if license_info == -1:
                raise BaseException()
            return license_info
        except:
            msg = "License check failed: Unable to communicate with service. Verify " + \
                  "Internet connection and try again or visit https://turi.com/support for support options."
            _mt._get_metric_tracker().track('get_license_info.config_parser_error')
            raise RuntimeError(msg)

    # online, offline-env, and offline-config all failed
    # error will be thrown from the unity server engine
    return None


def __set_license_info(product_key, license_info,
                       config_path=__default_config_path):
    """
    Sets the license info provided in file, which by default is ~/.graphlab/config
    Overwrites any existing product key in that file.

    Parameters
    ----------
    license_info : str
        The license info returned from the Turi server.

    config_path : str, optional
        Specifies which file to use for configuration (defaults to ~/.graphlab/config)
    """
    import graphlab.connect as _mt
    try:
        config = _ConfigParser.ConfigParser()
        config.read(config_path)
        if not(config.has_section(__section)):
            config.add_section(__section)
        config.set(__section, __key, product_key)
        config.set(__section, __license, license_info)
        with open(config_path, 'w') as config_file:
            config.write(config_file)
        _mt._get_metric_tracker().track('set_license_info.succeeded')
    except:
        _mt._get_metric_tracker().track('set_license_info.config_parser_error')
        __LOGGER__.warn('Unable to write current GraphLab Create license to %s. Ensure that this user account \
                        has write permission to %s to save the license for offline use.' % (config_path, config_path))
        _os.environ['GRAPHLAB_LICENSE_INFO'] = license_info
