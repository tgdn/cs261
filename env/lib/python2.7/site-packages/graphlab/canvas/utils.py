import graphlab.connect as _mt
import logging as _logging
import json as _json
import math as _math
import datetime as _datetime
import six as _six

import sys
if sys.version_info.major > 2:
    long = int

"""
This package contains the implementation of the server component of GraphLab
Canvas, the visualization engine for GraphLab Create data structures.
"""

__LOGGER__ = _logging.getLogger(__name__)
_active_target = None
_active_port = None

def __in_ipython_notebook():
    #check for any IPython installed. return true if it is version < 4.0

    # IPython.core.getipython.get_ipython() returns ipykernel.zmqshell.ZMQInteractiveShell
    # if set_target is called too early
    try:
        import IPython
        if isinstance( IPython.core.getipython.get_ipython(), \
                IPython.kernel.zmq.zmqshell.ZMQInteractiveShell):
            return True
    except:
        pass
    # check if it is Jupyter
    try:
        import ipykernel
        if isinstance( \
                IPython.core.getipython.get_ipython(), \
                ipykernel.zmqshell.ZMQInteractiveShell):
            return True
    except:
        return False

    return False

def set_target(target, port=None):
    """
    Set the target for GraphLab Canvas view output. Canvas will only be
    invoked once .show() has been called on a GLC object or
    via :func:`graphlab.canvas.show`

    Parameters
    ----------
    target: str, optional

        - 'browser': Calling .show() on a GLC object will start Canvas
          and open a web browser locally to the Canvas interface.

        - 'headless': Calling .show() on a GLC object will start Canvas
          without opening a local web browser.

        - 'ipynb': Calling .show() will attempt to render to an output cell
          in the IPython Notebook or Jupyter Notebook. Note: this target 
          requires an active internet connection from the GraphLab Create 
          instance to load resources from https://static.turi.com.

        - 'none': The Canvas interface will not be accessible in any manner.

    port: integer, optional

        a value between 1025 - 65535

    """
    from . import target as __target
    global _active_port
    global _active_target

    targets = {
        'browser': __target.InteractiveTarget,
        'headless': __target.HeadlessTarget,
        'ipynb': __target.IPythonTarget,
        'none': __target.NoneTarget
    }

    if not(target in targets):
        raise ValueError('Canvas target \'%s\' is an invalid value. ' \
            'Select from: %s.' % (target, list(targets.keys())))

    if target in ['browser', 'headless']:
        # for browser and headless, allow resetting target if the port changes
        if not(isinstance(_active_target, targets[target])) or \
           port != _active_port:
            try:
                _active_target = targets[target](port)
                _active_port = port
            except:
                print("Error: Requested port is unavailable: %s" % port)
    elif not(isinstance(_active_target, targets[target])):
        # for other targets, only set if the active target changed
        if target == 'ipynb' and not(__in_ipython_notebook()):
            __LOGGER__.warn("This Python session does not appear to be running in an interactive IPython Notebook or Jupyter Notebook. Use of the 'ipynb' target may behave unexpectedly or result in errors.")
            # we will set it anyway, in case this is really what the user wants.
            # and to preserve backwards compatibility/other use cases
            # (running a notebook exported to .py, for instance)
        _active_target = targets[target]()

    # track metrics on target
    _mt._get_metric_tracker().track('canvas.set_target.%s' % target)

def get_target():
    """
    Get the active target for Canvas. If none has been set, this will set
    the default target ("browser") as the active target and return it.
    """
    global _active_target
    if _active_target is None:
        set_target('browser')
    return _active_target


def show(variable=None):
    """
    show()
    Re-launch and resume the prior GraphLab Canvas session in default browser.
    This method is useful if the GraphLab Create Python session is still active
    but the GraphLab Canvas browser session has ended.

    """
    if variable is not None:
        get_target().state.set_selected_variable(variable)
        variable.validate_js_component_name(variable.get_js_component())
    return get_target().show()

def _get_id(ref):
    import graphlab
    import graphlab.data_structures.gframe
    import graphlab.toolkits._model
    # use real content identity of underlying SArray to
    # determine whether two are the same.
    # will err on the side of correctness (it's possible two
    # objects that really are the same seems different according
    # to this function) due to lazy evaluation behavior with
    # __get_content_identifier__.
    # for other known types (SFrame and SGraph) use proxy object.
    if isinstance(ref, graphlab.SArray):
        # use content identity (goes all the way to real storage, so
        # two C++ objects w/ two corresponding Python objects could
        # still be the same.
        return ref.__get_content_identifier__()

    if type(ref) == graphlab.SFrame or \
       type(ref) == graphlab.SGraph or \
       isinstance(ref, graphlab.toolkits._model.Model):
        return long(hash(str(ref.__proxy__)))

    if type(ref) == graphlab.data_structures.gframe.GFrame:
        # hash a tuple of SGraph proxy and GFrame type
        return long(hash((str(ref.__graph__.__proxy__), ref.__type__)))

    if isinstance(ref, (graphlab.deploy.environment._Environment,
                        graphlab.deploy._job.Job)):
        return long(hash(str(ref.name)))

    # for all other types, we don't really know how to compare
    # so just use the Python id
    return long(id(ref))

def _same_object(ref1, ref2):
    if type(ref1) != type(ref2):
        return False
    return _get_id(ref1) == _get_id(ref2)

def _profile(func):
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile, pstats, StringIO, tempfile
        import time
        pr = cProfile.Profile()
        pr.enable()
        time1 = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            time2 = time.time()
            pr.disable()
            s = StringIO.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(s.getvalue())
                print('function took %0.3f ms' % ((time2-time1)*1000.0))
                print('log file at %s' % f.name)
    return wrapper

def _to_json(value):
    # wrap JSON.dumps, uses CustomJSONEncoder logic to serialize
    return _json.dumps(_remove_nan(value), allow_nan=False, cls=_CustomJSONEncoder)

def _remove_nan(value):
    # replaces nans with None recursively to ensure JSON serializability
    if isinstance(value, dict):
        # recursively apply to dict values
        return {k: _remove_nan(v) for (k,v) in value.items()}
    elif isinstance(value, (list, tuple)):
        # recursively apply to list values
        return [_remove_nan(v) for v in value]
    elif isinstance(value, float) and \
         (_math.isnan(value) or _math.isinf(value)):
        # replace nan/inf with None
        return None
    else:
        # pass through, return original
        return value

class _CustomJSONEncoder(_json.JSONEncoder):
    """
    Custom JSON Encoding Logic.  Implements the default function called by
    json.dumps().  Only serializes object types not already encoded by
    JSONEncoder (None, Boolean, int, long, float, list, tuple, dict)
    """
    def default(self, obj):
        """
        This function is called by json.dumps on any value passed in.

        Parameters
        ----------
        obj: any

        Returns
        -------
        JSON Serializable object or None

        """
        if isinstance(obj, _datetime.datetime) or \
             isinstance(obj, _datetime.date):
            # replaces datetimes and dates with isoformat string
            return obj.isoformat()
        elif sys.version_info.major > 2 and isinstance(obj, map):
            return list(obj)
        else:
            # pass through, return original
            return super(_CustomJSONEncoder, self).default(obj)
