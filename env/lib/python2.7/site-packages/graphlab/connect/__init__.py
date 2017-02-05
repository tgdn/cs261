"""
This module defines classes and global functions for creating and managing
connection to the graphlab backend server.
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
# - The entire connect module is duplicated for simplicity.
# - OSS does not have product key checks
# ---------------------------------------------------------------------------



""" The module usage metric tracking object """
from graphlab.util.config import DEFAULT_CONFIG as _default_local_conf
from graphlab.util.metric_tracker import MetricTracker as _MetricTracker


""" The global client object """
__CLIENT__ = None

""" The global graphlab server object """
__SERVER__ = None

""" The module usage metric tracking object """
__USAGE_METRICS__ = _MetricTracker(mode=_default_local_conf.mode)

__UNITY_GLOBAL_PROXY__ = None

def _get_metric_tracker():
    """
    Returns the global metric tracker object.
    """
    return __USAGE_METRICS__
