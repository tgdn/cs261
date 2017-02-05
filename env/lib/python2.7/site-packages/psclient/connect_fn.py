import os as _os
from .predictive_service_client import PredictiveService as \
    _PredictiveServiceClient

import psclient
from .config import load_config

def connect(
        manager_endpoint=None,
        admin_key=None,
        query_endpoint=None,
        verify_certificate=None,
        query_timeout=None,
        name=None,
        config_file=None):

    params = load_config(
        name=name,
        config_file=config_file,
        params = {
            'manager_endpoint':manager_endpoint,
            'query_endpoint':query_endpoint,
            'admin_key':admin_key,
            'verify_certificate':verify_certificate,
        },
    )

    if not params['manager_endpoint'] or not params['admin_key']:
        raise ValueError("Cannot load PredictiveService client."
            " manager_endpoint and admin_key were not specified.")

    return _PredictiveServiceClient(**params)

connect.__doc__ = psclient.connect.__doc__
