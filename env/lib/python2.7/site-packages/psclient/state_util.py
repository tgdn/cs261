from ConfigParser import ConfigParser as _ConfigParser
import json as _json
from time import time as _time

# State File Config Section Names
_DEPLOYMENT_SECTION_NAME = 'Predictive Objects Service Versions'
_PREDICTIVE_OBJECT_DOCSTRING = 'Predictive Objects Docstrings'
_ENVIRONMENT_SECTION_NAME = 'Environment Info'
_SERVICE_INFO_SECTION_NAME = 'Service Info'
_SYSTEM_SECTION_NAME = 'System'
_ENVIRONMENT_VARIABLE_SECTION_NAME = 'Environment Variables'
_META_SECTION_NAME = 'Meta'

def generate_init_state_file(name, environment, description, api_key,
                             admin_key, aws_credentials, cors_origin,
                             global_cache_state, system_config):
    '''
    Initialize a new Predictive Service state file and save to s3/hdfs/local.
    '''
    # Construct a few states
    all_keys = {}
    all_keys[api_key] = {"id": api_key, "type": "api", "endpoints": None,
                         "adhoc_execute": False}
    all_keys[admin_key] = {"id": admin_key, "type": "admin"}
    scale_factor = 1
    revision_number = 1

    # Dump state data to a config
    state = _ConfigParser(allow_no_value=True)
    state.optionxform = str
    state.add_section(_SERVICE_INFO_SECTION_NAME)
    state.set(_SERVICE_INFO_SECTION_NAME, 'Name', name)
    state.set(_SERVICE_INFO_SECTION_NAME, 'Creation Time', int(_time()))
    state.set(_SERVICE_INFO_SECTION_NAME, 'Description', description)
    state.set(_SERVICE_INFO_SECTION_NAME, 'Keys', _json.dumps(all_keys))
    state.set(_SERVICE_INFO_SECTION_NAME, 'CORS Origin', cors_origin)
    state.set(_SERVICE_INFO_SECTION_NAME, 'Global Cache State', global_cache_state)
    state.set(_SERVICE_INFO_SECTION_NAME, 'Global Scale Factor', scale_factor)

    # Save environment
    if environment:
        state.add_section(_ENVIRONMENT_SECTION_NAME)
        for (key, value) in environment._get_state().iteritems():
            state.set(_ENVIRONMENT_SECTION_NAME, key, value)

    # initialize deployment version data to config
    state.add_section(_DEPLOYMENT_SECTION_NAME)
    state.add_section(_PREDICTIVE_OBJECT_DOCSTRING)

    # Update the revision number after we have successfully written all endpoints
    revision_number += 1
    state.add_section(_META_SECTION_NAME)
    state.set(_META_SECTION_NAME, 'Revision Number', revision_number)

    # Save system config. params
    state.add_section(_SYSTEM_SECTION_NAME)
    system_config.set_state_in_config(
        state, _SYSTEM_SECTION_NAME)

    # Add environment variables section
    state.add_section(_ENVIRONMENT_VARIABLE_SECTION_NAME)

    return state
