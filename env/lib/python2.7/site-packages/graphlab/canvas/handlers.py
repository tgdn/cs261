"""
Methods to be used internally by graphlab.canvas.server to handle HTTP requests.
"""

import graphlab
import graphlab.connect as _mt
from graphlab.canvas import statichandler as static
from graphlab.canvas.log_stream_handler import LogSSEHandler
from graphlab.canvas.utils import _to_json

import json
import os
import sys
import tornado.web

def __get_webapp_root_dir():
    """
    Absolute path to webapp static resources
    """
    relative_path = '/webapp/'
    return os.path.dirname(__file__) + relative_path

def __error_handler(handler):
    """
    A Tornado handler that does nothing and returns {},
    for use as a fallback when errors occur during request handling.
    """
    handler.set_status(500)
    handler.write({})

def __make_handler(fn, state, server, *args, **kwargs):
    """
    Allows partial function application in a way that is compatible with Tornado handlers.
    Using functools.partial does not work with instance methods for some reason, but we can
    pass functions wrapped in __make_handler as instance methods and it works.
    Also does exception handling through state.add_exception to get exception messages to the UI.
    """
    def applied(*b, **c):
        server.ping()
        d = {}
        d.update(kwargs)
        d.update(c)
        try:
          return fn(state, *(args + b), **d)
        except:
          e = sys.exc_info()
          state.add_exception(e)
          return __error_handler(b[0]) # assumes the first argument is 'handler'
    return applied

def objectSpecificRoute(state, method, handler, path):
    view = state.get_selected_variable()[1]
    view._handle_request(method, path, handler)

def expand_selected_variable(state):
    # get descriptives for variable
    selected_variable = state.get_selected_variable()
    if selected_variable is None:
        return None
    name, var = selected_variable
    if var is None:
        return None
    data = {
        'name': name,
        'type': var.objectType,
        'view_file': var.get_js_file(),
        'view_component': var.get_js_component(),
        'view_components': var.get_js_components()
    };
    data.update(var.get_metadata())
    return data

def __ping(state, server, handler):
    """
    Used when the window is not focused to just send an alive ping
    """
    server.ping()

def __get_var(state, server, handler):
    """
    Gives the browser data about Python variables, columns, selected state, etc.
    """
    server.ping()
    data = {
        'variables': state.get_variables(),
        'selected_variable': expand_selected_variable(state),
        'exceptions': state.get_exceptions(),
        'gl_version': graphlab.version,
        'gl_product_key': graphlab.product_key.get_product_key()
    }
    handler.write(data)

def __post_var(state, server, handler):
    """
    Sets the selected variable
    """
    selected_var = handler.get_argument('selected')
    selected_var = json.loads(selected_var) # json decode->list
    if type(selected_var) == list:
        selected_var = list(map(lambda x: x.encode('utf-8').decode(), selected_var)) # convert unicode
        selected_var = tuple(selected_var) # list->tuple
    else:
        selected_var = selected_var.encode('utf-8').decode()
    state.set_selected_variable(selected_var)
    # send the global state (selection info) back to the browser
    __get_var(state, server, handler)

def __post_metrics(state, handler):
    """
    Allows the posting of metrics from JavaScript through our usual Python code.
    """
    kwargs = {
        'event_name': 'canvas.js.%s' % handler.get_argument('metric'),
        'value': int(handler.get_argument('value', default='1')),
        'properties': json.loads(handler.get_argument('properties', default="{}"))
    }
    _mt._get_metric_tracker().track(**kwargs)

class JSONResponseHandler(tornado.web.RequestHandler):
    def write(self, chunk):
        """
        Make sure we write chunks as valid JSON (by default Tornado does not -- it includes Infinity and NaN)
        """
        if isinstance(chunk, dict):
            chunk = _to_json(chunk)
            self.set_header("Content-Type", "application/json; charset=UTF-8")
        super(JSONResponseHandler, self).write(chunk)

__handler_class_id = 0
def __make_handler_class(url, get=None, post=None):
    """
    Creates a tornado.web.RequestHandler class from the given get and post handlers.
    """
    global __handler_class_id
    handlerFunctions = {}
    if get is not None:
        handlerFunctions['get'] = get
    if post is not None:
        handlerFunctions['post'] = post
    __handler_class_id += 1
    return (url, type('CanvasHandler%d' % __handler_class_id, (JSONResponseHandler,), handlerFunctions))

def __get_static_handlers():
    """
    Get a list of handlers for static files as tornado.web.RequestHandler instances.
    """
    return [("/(.*\\.%s)" % ext, static.Handler, {'path': __get_webapp_root_dir()}) for ext in \
      ['css',
       'html',
       'ico',
       'jpg',
       'js',
       'png',
       'woff',
       'ttf',
       'eot',
       'otf',
       'svg']]

def __get_log_stream_handler():
    return [(r"/stream/(.*)", LogSSEHandler)]

def get_handlers(server, state):
    """
    Get a list of handlers as tornado.web.RequestHandler instances.
    """
    ret = [
        __make_handler_class(
            url=r"/ping",
            get=__make_handler(__ping, state, server, server)),
        __make_handler_class(
            url=r"/vars",
            get=__make_handler(__get_var, state, server, server),
            post=__make_handler(__post_var, state, server, server)),
        __make_handler_class(
            url=r"/metrics",
            post=__make_handler(__post_metrics, state, server)),
        (r"/(index\.html)", static.Handler, {'path': __get_webapp_root_dir()})
    ]
    ret.extend(__get_static_handlers())
    ret.extend(__get_log_stream_handler())
    ret.extend([
        __make_handler_class(
            url=r"/(.*)",
            get=__make_handler(objectSpecificRoute, state, server, 'get'),
            post=__make_handler(objectSpecificRoute, state, server, 'post'))
    ])
    return ret
