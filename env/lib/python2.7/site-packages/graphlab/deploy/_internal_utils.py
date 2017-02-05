import os
import graphlab as _gl
from . import _artifact
import logging
import xml.etree.ElementTree as oak_tree

import sys
if sys.version_info.major == 2:
    from urlparse import urlparse
else:
    import urllib.parse as urlparse  # Python 3

__LOGGER__ = logging.getLogger(__name__)

def copy_attributes(obj_to, obj_from, lst = None):
    """
    Copy all the atributes (by reference) from one object to another."

    Parameters
    ----------
    obj_to    : Copy to this object.
    obj_from  : Copy from this object.
    lst       : Copy these atrributes. If set to None, copy all.

    Notes
    -----
    Do not copy over things blindly. The None option could result in potentially
    very dangerous outcomes.

    """
    lst = dir(obj_from) if not lst else lst
    for attr in lst:
        obj_to.__setattr__(attr, getattr(obj_from, attr))


def load_gl_object(filename):
    """
    Load a GLC datastructure from a filename.

    Parameters
    ----------
    filename : Filename for the archive

    Returns
    ----------
    The GLC object.

    """
    obj = None
    if not os.path.exists(filename):
        raise "Loading error: %s is not a valid filename." % filename

    try:
        obj = _gl.load_sframe(filename)
        return obj
    except:
        pass
    try:
        obj = _gl.load_sgraph(filename)
        return obj
    except:
        pass

    try:
        obj = _gl.load_model(filename)
        return obj
    except:
        pass

    try:
        obj = _gl.SArray(data=filename)
        return obj
    except:
        pass

    return obj

def is_gl_artifact_class(obj_class):
    """
    Check if class is a GraphLab artifact.

    The function does it by checking the method resolution order (MRO) of the
    class and verifies that _Artifact is the base class.

    Parameters
    ----------
    obj_class    : Class to be checked.

    Returns
    ----------
    True if the class is a GLC Artifact.

    """
    # If it has no base classes, then it is not a GLC Artifact
    if not hasattr(obj_class, '__bases__'):
        return False

    # Check if _artifact.artifact is a base class
    return issubclass(obj_class, _artifact.Artifact)

def is_gl_model_class(obj_class):
    """
    Check if class is a GraphLab create model.

    The function does it by checking the method resolution order (MRO) of the
    class and verifies that _Model is the base class.

    Parameters
    ----------
    obj_class    : Class to be checked.

    Returns
    ----------
    True if the class is a GLC Model.

    """
    # If it has no base classes, then it is not a GLC Model.
    if not hasattr(obj_class, '__bases__'):
        return False

    # Check if _model.CustomModel is a base class
    return issubclass(obj_class, _gl.toolkits._model.CustomModel)

def is_gl_class(obj_class):
    """
    Check if class is a GraphLab create class.

    A GLC class is either a GLC data structure class (SFrame, SGraph etc.) or
    a GLC model class.

    Parameters
    ----------
    obj_class    : Class to be checked.

    Returns
    ----------
    True if the class is a GLC class.

    """
    # GLC-Data structures
    gl_ds = [_gl.SFrame, _gl.SArray, _gl.SGraph]

    # Object is GLC-DS or GLC-Model
    return (obj_class in gl_ds) or is_gl_model_class(obj_class)

def read_hadoop_config(conf_path):
    if os.path.exists(conf_path):
        try:
            tree = oak_tree.parse(conf_path)
        except:
            __LOGGER__.error("Unable to parse %s" % conf_path)
            return
        root = tree.getroot()
        for p in root.findall("./property"):
            yield p


def read_core_config(core_site_path):

    config = []
    core_site = core_site_path + '/core-site.xml'
    for property in read_hadoop_config(core_site):

        # fs.default.name is the key name for the file system on EMR clusters
        if property.findall('name')[0].text in ('fs.defaultFS', 'fs.default.name'):
            parse_result = urlparse(property.findall('value')[0].text)
            __LOGGER__.debug("Got namenode '%s' from %s" % (parse_result.geturl(), core_site))

            config.append({"namenode": parse_result.hostname,
                           "port": parse_result.port if parse_result.port
                           else 8020})

    return config

def find_core_hadoop_config(core_hadoop_dir):
    if not core_hadoop_dir:
        try_paths = ['/etc/hadoop/conf',
                      '/usr/local/etc/hadoop/conf',
                      '/usr/local/hadoop/conf']

        if 'HADOOP_CONF_DIR' in os.environ:
            try_paths.insert(0, os.environ['HADOOP_CONF_DIR'])

    else:
        core_hadoop_dir = os.path.abspath(os.path.expanduser(core_hadoop_dir))
        try_paths = [core_hadoop_dir]

    for path in try_paths:
        __LOGGER__.info('Attempting to find hadoop core-site.xml at %s' % path)
        configs = read_core_config(path)
        if len(configs) > 0:
            __LOGGER__.info( 'configs = %s ' % configs)
            return configs

    raise RuntimeError('Cannot find hadoop configuration, please specify a value '
        'for "hadoop_conf_dir" when creating Hadoop Cluster, or set the '
        'HADOOP_CONF_DIR environment variable.')


