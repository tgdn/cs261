"""
This module contains the main logic for start, query, stop graphlab server client connection.
"""

'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''

#
# ---------------------------------------------------------------------------
# THIS IS AN OSS OVERRIDE FILE
#
# What this means is that there is a corresponding file in the OSS directory,
# and this file overrides that. Be careful when making changes.
# Specifically, do log the differences here.
#
# - OSS does not have product key checks
# - OSS does not have clear_distributed_execution_environment
# - Engine start error message is different (directs to GitHub issues)
# ---------------------------------------------------------------------------

from ..cython.cy_unity import UnityGlobalProxy
from ..cython import cy_ipc
from ..cython.cy_server import EmbeddedServer
from ..connect import __SERVER__, __CLIENT__
from .. import product_key as _product_key

import logging
import os

""" The module level logger object """
__LOGGER__ = logging.getLogger(__name__)

""" Global varialbes """
__UNITY_GLOBAL_PROXY__ = None

ENGINE_START_ERROR_MESSAGE = 'Cannot start GraphLab Create engine.'


def launch(server_addr=None, server_bin=None,
           server_log=None, auth_token=None, server_public_key=''):
    """
    Launch a connection to the graphlab server. The connection can be stopped by
    the `stop` function.

    Automatically spawns a local server, if no arguments provided or "server_bin"
    is specified.

    Notes
    -----
        Only a single connection can exist at anytime.
        Prints warning if trying to launch while there is an active connection.

    Parameters
    ----------
    server_addr : string
        The address of the server.

    server_bin : string
        The path to the server binary (local server only).

    server_log : string
        The path to the server log (local server only).

    server_public_key : string
        The server's libsodium public key, used for encryption. Default is no encryption.
    """
    if is_connected():
        __LOGGER__.warning(
            "Attempt to connect to a new server while still connected to a server."
            " Please stop the connection first by running 'graphlab.stop()' and try again.")
        return

    # construct the server instance
    if server_addr is None:
        server_addr = 'inproc://sframe_server'

    # Good to go
    server = None
    try:
        server = EmbeddedServer(server_addr, server_log)
        _raise_if_invalid_product_key_or_license(server)
        server.start()
    except Exception as e:
        if server:
            server.try_stop()
        raise e
    server.set_log_progress(True)
    # start the client
    client = cy_ipc.make_comm_client_from_existing_ptr(server.get_client_ptr())

    _assign_server_and_client(server, client)

    assert is_connected()


def stop():
    """
    Stops the current connection to the graphlab server.
    All object created to the server will be inaccessible.

    Reset global server and client object to None.
    """
    global __CLIENT__, __SERVER__
    if not is_connected():
        return
    if (__CLIENT__):
        __CLIENT__.stop()
        __CLIENT__ = None
    if (__SERVER__):
        __SERVER__.try_stop()
        __SERVER__ = None


def is_connected():
    """
    Returns true if connected to the server.
    """
    if (__CLIENT__ is not None and __SERVER__ is not None):
        # both client and server are live
        return True
    elif (__CLIENT__ is None and __SERVER__ is None):
        # both client and server are dead
        return False
    else:
        # unlikely state: one of them are live and the other dead
        raise RuntimeError('GraphLab connection error.')


def get_client():
    """
    Returns the global ipc client object, or None if no connection is present.
    """
    if not is_connected():
        launch()
    assert is_connected(), ENGINE_START_ERROR_MESSAGE
    return __CLIENT__


def get_server():
    """
    Returns the global graphlab server object, or None if no connection is present.
    """
    if not is_connected():
        launch()
    assert is_connected(), ENGINE_START_ERROR_MESSAGE
    return __SERVER__


def get_unity():
    """
    Returns the unity global object of the current connection.
    If no connection is present, automatically launch a localserver connection.
    """
    if not is_connected():
        launch()
    assert is_connected(), ENGINE_START_ERROR_MESSAGE
    return __UNITY_GLOBAL_PROXY__


def _assign_server_and_client(server, client):
    """
    Helper function to assign the global __SERVER__ and __CLIENT__ pair.
    """
    global __SERVER__, __CLIENT__, __UNITY_GLOBAL_PROXY__
    __SERVER__ = server
    __CLIENT__ = client
    __UNITY_GLOBAL_PROXY__ = UnityGlobalProxy(__CLIENT__)
    server.get_logger().info('GraphLab Create v%s started. Logging: %s' %
                             (UnityGlobalProxy(client).get_version(), server.unity_log))

    from ..extensions import _publish
    _publish()


class InvalidProductKey(Exception):
    pass


class InvalidLicense(Exception):
    pass


def _raise_if_invalid_product_key_or_license(server):
    default_config_path = _product_key.__default_config_path
    if 'GRAPHLAB_DEFAULT_CONFIG_LOCATION' in os.environ:
        default_config_path = os.environ['GRAPHLAB_DEFAULT_CONFIG_LOCATION']
    # Fetch product key
    try:
        product_key = _product_key.get_product_key(default_config_path)
        server.product_key = product_key
    except KeyError as k:
        raise InvalidProductKey(str(k))

    # check product key
    if not server._is_product_key_valid():
        if server.product_key is None:
            __LOGGER__.error(" ========================================\n"
                             "GraphLab Create requires a license to use. To get a non-commercial "
                             " license for academic use only, visit https://turi.com/register.\n"
                             "=================================================\n")
            raise InvalidProductKey("Product key not found.")
        else:
            raise InvalidProductKey("Product key validation failed.")

    # Fetch license key
    try:
        license_info = _product_key._get_license_info(default_config_path)
        server.license_info = license_info
    except Exception as k:
        raise InvalidLicense(k.args[0])

    # check license info
    if not server._is_license_valid():
        message = server._get_license_message()
        if len(message) > 0:
            print(message)
        raise InvalidLicense("License check failed.")
    else:
        print(server._get_license_message())

# Register an exit callback handler to stop the server on python exit.
import atexit
atexit.register(stop)
