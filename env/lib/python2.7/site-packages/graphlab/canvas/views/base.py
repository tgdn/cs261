import array
import datetime
import re
import textwrap
import six
import sys

import graphlab
from graphlab.canvas.utils import _to_json

import sys
if sys.version_info.major == 3:
    long = int


def _unbox_data_structure(struct):
    """
    SFrame and SArrays are unboxed into nested lists/dicts.
    Dictionaries and lists of SFrames and SArrays are acceptable.
    Operates recursively.

    Parameters
    ----------
    value: list, dict, SFrame, SArray
         object to be recurisvly encoded into JSON

    Returns
    -------
    json_string: str
     data structure (SFrame, SArray, SGraph) cast into dict/list and dumped to JSON

    """

    if isinstance(struct, list):
        return [ _unbox_data_structure(item) for item in struct ]
    elif isinstance(struct, graphlab.data_structures.sframe.SFrame):
        return list(struct.pack_columns(column_prefix='', dtype=dict)['X1'])
    elif isinstance(struct, graphlab.data_structures.sframe.SArray):
        return list(struct)
    elif isinstance(struct, graphlab.data_structures.sgraph.SGraph):
        #TODO implement for SGraph
        raise NotImplementedError('SGraph not supported.')
    #encode_value will truncate and json-ify lists/dicts.  Need to treat them differently
    elif isinstance(struct, dict):
        return { k: _unbox_data_structure(v) for (k, v) in six.iteritems(struct) }
    else:
        return _encode_value(struct)


def _encode_value(value):
    """
    This will truncate at 24 characters with ellipses as necessary, will format
    list/dict/array as JSON before truncating, and will replace bad characters
    in non-utf8 strings.

    Parameters
    ----------
    value: string, float, long, int, datetime, list, dict, Image
         object to be recurisvly encoded into JSON

    Returns
    -------
    plain_object: list, dict
         mixed object with all SFrames and SArrays cast into list/dicts

    Called to encode a single value (at row/col) from the SFrame
    or SArray as a JSON string. This will truncate at 24 characters
    with ellipses as necessary, will format list/dict/array as JSON
    before truncating, and will replace bad characters in non-utf8 strings.
    """
    # leave numbers alone
    if isinstance(value, (int, long, float)):
        return value

    # leave Nones alone (they turn into null in JSON)
    if value is None:
        return value

    # convert datetime to str
    if isinstance(value, datetime.datetime):
        # return, don't go through truncation
        return str(value)

    # represent image as base64-encoded bytes
    import binascii
    from graphlab.data_structures.image import Image
    if isinstance(value, Image):
        image_format = None
        if value._format_enum == 0:
            image_format = 'jpeg'
        elif value._format_enum == 1:
            image_format = 'png'
        elif value._format_enum == 2:
            image_format = 'raw'
        if image_format is not None:
            ret = {
                'type': 'image',
                'width': value._width,
                'height': value._height,
                'channels': value._channels,
                'format': image_format,
                'id': id(value)
            }
            if image_format in ('jpeg', 'png'):
                ret.update({
                    'value': 'image/%s;base64,%s' % (image_format, binascii.b2a_base64(value._image_data))
                })
            elif image_format == 'raw':
                ret.update({
                    'value': list(value._image_data)
                })
            return ret

        # fallback case for images the browser does not know how to display
        # just convert to str and treat like any other type
        value = str(value)

    # convert strings to unicode (assumes utf-8 encoding, replaces invalid
    # characters with ?
    if isinstance(value, str) and sys.version_info.major == 2:
        value = unicode(value, encoding='utf-8', errors='replace')

    # get the array into a list so it is JSON serializable
    if isinstance(value, array.array):
        value = value.tolist()

    # truncate to 10 elements first
    if isinstance(value, (array.array, list)):
        value = value[:10]
    elif isinstance(value, dict):
        keys = value.keys()[:10]
        truncated = {}
        for key in keys:
            truncated[key] = value[key]
        value = truncated

    # get dict/list values properly encoded inside before dumping to str
    if isinstance(value, list):
        value = [_encode_value(v) for v in value]
    elif isinstance(value, dict):
        value = {_encode_value(k): _encode_value(v) for (k,v) in six.iteritems(value)}

    # json serialize dict/list types to convert to string
    if isinstance(value, (dict, list)):
        value = _to_json(value)

    # truncate via textwrap (will break on word boundaries if possible)
    wrapped = textwrap.wrap(value, 18)
    if len(wrapped) == 0:
        return ''

    return '%s%s' % (
        wrapped[0],
        '' if len(wrapped) == 1 else ' ...'
        )

class BaseView(object):
    """
    Base MVC view component for Canvas. Inherit from this to define specific
    behavior in Canvas for an object (data structure or GL Model).
    """
    def __init__(self, obj, params=None):
        self.obj = obj
        self.objectType = type(obj).__name__
        self.handlers = {}
        self.children = {}
        self.params = {} if params is None else params

    def _handle_request(self, method, url, handler):
        """
        This method does URL routing for registered handlers (those
        that have been added with register_handler) on this view.
        You do not need to override this method to create a new View object
        -- just register handlers in __init__ with appropriate methods and paths.

        This method will be called for each HTTP request by Tornado.

        Parameters:
        -----------
        method : String
          The HTTP method of the request. ('get', 'post', etc.)

        path : String
          The server-relative URL path of the request.

        handler : tornado.web.RequestHandler
          The Tornado request handler (use methods on this to send a response).
        """
        if not(method in self.handlers):
            handler.set_status(405) # Method Not Allowed
            handler.write({})
            return
        for (path, fn) in self.handlers[method].items():
            if re.match(path, url):
                fn(url, handler)
                return
        handler.set_status(404) # Not Found
        handler.write({})

    def _in_ipython(self):
        """
        This method determines whether Canvas is targetting IPython
        """
        import graphlab.canvas.target
        return isinstance(graphlab.canvas.get_target(), graphlab.canvas.target.IPythonTarget)


    def register_handler(self, method, path, fn):
        """
        Call this (in __init__) to register a request handler function.
        The signature of that function should take two parameters
        (url and tornado.web.RequestHandler) and return None (or have no return
        statement). The fn should respond by calling methods on the
        tornado.web.RequestHandler like set_status and/or write.

        Parameters:
        -----------
        method : String
          The HTTP method (like 'get' or 'post') to handle.

        path : String
          A regular expression matching the server-relative URLs that
          should be handled with this handler function.

        fn : Function
          A function whose signature is (self, tornado.web.RequestHandler ->
          None) that handles the request by writing to the tornado.web.RequestHandler.
        """
        if not(method in self.handlers):
            self.handlers[method] = {}
        self.handlers[method][path] = fn

    def child_views(self):
        """
        Override this method to return a dictionary of child views (navigable
        elements in Canvas that are scoped to this object). Any view returned
        in this dictionary will have the navigation breadcrumb point back to this
        object when active. The format of the dictionary should be: {name: view}.

        Example: for an SFrame with columns 'foo' and 'bar', this will return:
        { 'foo': SArrayView, 'bar': SArrayView }

        Views are cheap to instantiate but if they do any internal caching (like
        SArrayView) then the object should hold onto child view references rather
        than regenerating them on each call to this method (see SFrameView.py)
        or else it will cause extra work (more object instantiation on each request).
        """
        return self.children

    def get_metadata(self):
        """
        Override this method to return a dictionary representation of the
        underlying data from this object (whatever the corresponding JavaScript
        view will need to render). The dictionary returned by this method
        will be merged with one containing the keys 'name' and 'type', so do
        not use those keys.

        In general this operation should be cheap (nearly free) -- small data
        ( < 1 KB) and very little processing time. This method will be called
        frequently to see if the data (or selected variable) has changed.

        If you want to do a long-running operation (utilizing the progress bar
        in Canvas) use the handle_request method instead.
        """
        return {}

    def get_staticdata(self):
        """
        Override this method to return a full dictionary representation of the
        underlying data from this object. This should be (in general) an
        amalgamation of get_metadata and any REST GET APIs exposed by this
        object. All data will be supplied to the JavaScript view simultaneously.

        This method will be called when generating static output in IPython
        Notebook (suitable for conversion to HTML with nbconvert).
        """
        return self.get_metadata()

    def get_js_file(self):
        """
        Override this method to provide a path to a JS file that
        will render this View in the browser. This method should return a str
        representation of the file path.

        The JS file specified must follow the requirejs AMD module specification:
        http://requirejs.org/docs/api.html#modulename
        """
        return 'placeholder'

    def get_js_component(self):
        """
        Override this method to provide a component name that
        will render this View in the browser. This method should return a str
        representation of the component name in the AMD module.

        The JS component currently must be a React.js component:
        http://facebook.github.io/react/docs/top-level-api.html#react.createclass
        """
        return 'View';

    def get_js_components(self):
        """
        Override this method to provide a list of possible components.
        Each will be shown as a tab to the user.
        The default is to show just the selected component.
        """
        return [self.get_js_component()]

    def get_temporary_name(self):
        """
        Override this method to provide a specific temporary name (tuple)
        for the View object that will be rendered in the browser. This method will be
        called when the state cannot find the name of the selected variable.
        """
        return ('<%s>' % type(self.obj).__name__,)

    def validate_js_component_name(self, name):
        """
        Override to raise an exception if an invalid view name is passed.
        """
        if not(name in self.get_js_components()):
            raise ValueError('View "%s" does not exist on this object.' % name)
