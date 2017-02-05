"""@package graphlab.toolkits

Defines a basic interface for a model object.
"""
'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''
import json

import graphlab as _gl
import graphlab.connect as _mt
import graphlab.connect.main as glconnect
from graphlab.data_structures.sframe import SFrame as _SFrame
from graphlab.toolkits._internal_utils import _toolkit_serialize_summary_struct
from graphlab.util import _make_internal_url
from graphlab import _gl_pickle as gl_pickle
from graphlab.toolkits._model_workflow import _ModelWorkflow
from graphlab.toolkits._main import ToolkitError
import graphlab.util.file_util as file_util
import os
from copy import copy as _copy

# ---------------------------------------------------------------------------
# THIS IS AN OSS OVERRIDE FILE
#
# What this means is that there is a corresponding file in the OSS directory,
# and this file overrides that. Be careful when making changes.
# Specifically, do log the differences here.
#
# - OSS does not have model workflow
# ---------------------------------------------------------------------------

def load_model(location):
    """
    Load any GraphLab Create model that was previously saved.

    This function assumes the model (can be any model) was previously saved in
    GraphLab Create model format with model.save(filename).

    Parameters
    ----------
    location : string
        Location of the model to load. Can be a local path or a remote URL.
        Because models are saved as directories, there is no file extension.

    Examples
    ----------
    >>> model.save('my_model_file')
    >>> loaded_model = gl.load_model('my_model_file')
    """
    _mt._get_metric_tracker().track('toolkit.model.load_model')

    # Check if the location is a dir_archive, if not, use glunpickler to load
    # as pure python model

    # We need to fix this sometime, but here is the explanation of the stupid
    # check below:
    #
    # If the location is a http location, skip the check, and directly proceed
    # to load model as dir_archive. This is because
    # 1) exists() does not work with http protocol, and
    # 2) GLUnpickler does not support http
    protocol = file_util.get_protocol(location)
    dir_archive_exists = False
    if protocol == '':
        model_path = file_util.expand_full_path(location)
        dir_archive_exists = file_util.exists(os.path.join(model_path, 'dir_archive.ini'))
    else:
        model_path = location
        if protocol in ['http', 'https']:
            dir_archive_exists = True
        else:
            import posixpath
            dir_archive_exists = file_util.exists(posixpath.join(model_path, 'dir_archive.ini'))

    if not dir_archive_exists:
        # Not a ToolkitError so try unpickling the model.
        unpickler = gl_pickle.GLUnpickler(location)

        # Get the version
        version = unpickler.load()

        # Load the class name.
        cls_name = unpickler.load()
        cls = _get_class_from_name(cls_name)

        # Load the object with the right version.
        model = cls._load_version(unpickler, version)

        unpickler.close()

        # Return the model
        return model
    else:
        _internal_url = _make_internal_url(location)
        return glconnect.get_unity().load_model(_internal_url)


def _get_default_options_wrapper(unity_server_model_name,
                                module_name='',
                                python_class_name='',
                                sdk_model = False):
    """
    Internal function to return a get_default_options function.

    Parameters
    ----------
    unity_server_model_name: str
        Name of the class/toolkit as registered with the unity server

    module_name: str, optional
        Name of the module.

    python_class_name: str, optional
        Name of the Python class.

    sdk_model : bool, optional (default False)
        True if the SDK interface was used for the model. False otherwise.

    Examples
    ----------
    get_default_options = _get_default_options_wrapper('classifier_svm',
                                                       'svm', 'SVMClassifier')
    """
    def get_default_options_for_model(output_type = 'sframe'):
        """
        Get the default options for the toolkit
        :class:`~graphlab.{module_name}.{python_class_name}`.

        Parameters
        ----------
        output_type : str, optional

            The output can be of the following types.

            - `sframe`: A table description each option used in the model.
            - `json`: A list of option dictionaries suitable for JSON serialization.

            | Each dictionary/row in the dictionary/SFrame object describes the
              following parameters of the given model.

            +------------------+-------------------------------------------------------+
            |      Name        |                  Description                          |
            +==================+=======================================================+
            | name             | Name of the option used in the model.                 |
            +------------------+---------+---------------------------------------------+
            | description      | A detailed description of the option used.            |
            +------------------+-------------------------------------------------------+
            | type             | Option type (REAL, BOOL, INTEGER or CATEGORICAL)      |
            +------------------+-------------------------------------------------------+
            | default_value    | The default value for the option.                     |
            +------------------+-------------------------------------------------------+
            | possible_values  | List of acceptable values (CATEGORICAL only)          |
            +------------------+-------------------------------------------------------+
            | lower_bound      | Smallest acceptable value for this option (REAL only) |
            +------------------+-------------------------------------------------------+
            | upper_bound      | Largest acceptable value for this option (REAL only)  |
            +------------------+-------------------------------------------------------+

        Returns
        -------
        out : dict/SFrame

        See Also
        --------
        graphlab.{module_name}.{python_class_name}.get_current_options

        Examples
        --------
        .. sourcecode:: python

          >>> import graphlab

          # SFrame formatted output.
          >>> out_sframe = graphlab.{module_name}.get_default_options()

          # dict formatted output suitable for JSON serialization.
          >>> out_sframe = graphlab.{module_name}.get_default_options('json')
        """
        _mt._get_metric_tracker().track('toolkit.%s.get_default_options' % module_name)
        if sdk_model:
            response = _gl.extensions._toolkits_sdk_get_default_options(
                                                          unity_server_model_name)
        else:
            response = _gl.extensions._toolkits_get_default_options(
                                                          unity_server_model_name)

        if output_type == 'json':
          return response
        else:
          json_list = [{'name': k, '': v} for k,v in response.items()]
          return _SFrame(json_list).unpack('X1', column_name_prefix='')\
                                   .unpack('X1', column_name_prefix='')

    # Change the doc string before returning.
    get_default_options_for_model.__doc__ = get_default_options_for_model.\
            __doc__.format(python_class_name = python_class_name,
                  module_name = module_name)
    return get_default_options_for_model


def _get_class_full_name(obj):
    """
    Returns a full class name from the module name and the class name.

    Parameters
    ----------
    cls : type | object

    Returns
    -------
    A full name with all the imports.
    """
    cls_name = obj.__name__ if type(obj) == type else obj.__class__.__name__
    return "%s.%s" % (obj.__module__ , cls_name)


def _get_class_from_name(class_name, *arg, **kwarg):
    """
    Create a class instance given the name of the class. This will ensure all
    the required modules are imported.

    For example. The class graphlab.my_model will first import graphlab
    and then load my_model.

    """
    module_path = class_name.split('.')
    import_path = module_path[0:-1]
    module = __import__('.'.join(import_path), fromlist=[module_path[-1]])
    class_ = getattr(module, module_path[-1])
    return class_

class CustomModel(object):
    """
    This class defines the minimal interface of a model that can be interfaced
    with GraphLab objects. This class contains serialization routines that make
    it compatible with GraphLab objects.

    Examples
    ----------
    # Define the model
    class MyModel(CustomModel):
        def __init__(self, sf, classifier):
            self.sframe = sf
            self.classifier = classifier

        def my_func(self):
            return self.classifier.predict(self.sframe)

    # Construct the model
    >>> custom_model = MyModel(sf, glc_model)

    ## The model can be saved and loaded like any GraphLab Create model.
    >>> model.save('my_model_file')
    >>> loaded_model = gl.load_model('my_model_file')
    """

    _PYTHON_MODEL_VERSION = 0

    def __init__(self, proxy = None):
        """
        Construct a dummy object from its proxy object and class name. The
        proxy contains a location to the pickle-file where the real object
        is stored.

        Parameters
        ----------
        proxy  : object

        Returns
        -------
        Returns an object of type _class with a path to the pickle archive
        saved in proxy.temp_file.

        """
        self.__proxy__ = None

    @classmethod
    def _is_gl_pickle_safe(cls):
        """
        Return True if the model is GLPickle safe i.e if the model does not
        contain elements that are written using Python + GraphLab objects.
        """
        return True

    def save(self, location):
        """
        Save the model. The model is saved as a directory which can then be
        loaded using the :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        location : string
            Target destination for the model. Can be a local path or remote URL.

        See Also
        ----------
        graphlab.load_model

        Examples
        ----------
        >>> model.save('my_model_file')
        >>> loaded_model = gl.load_model('my_model_file')

        """
        # Save to a temoporary pickle file.
        try:
            self._save_to_pickle(location)
        except IOError as err:
            raise IOError("Unable to save model. Trace (%s)" % err)

    def _save_to_pickle(self, filename):
        """
        Save the object to a pickle file.

        Parameters
        ----------
        filename : Filename to save.

        Notes
        -----
        The file is saved to a GLPickle archive. The following three attributes
        are saved:

        - The version of the object (obtained from get_version())
        - Class name of the object.
        - The object is pickled as directed in _save_impl

        """
        if not self._is_gl_pickle_safe():
            raise RuntimeError("Cannot pickle object. Use the 'save' method.")

        # Setup the pickler.
        pickler = gl_pickle.GLPickler(filename)

        # Save version and class-name.
        pickler.dump(self._get_version())
        pickler.dump(_get_class_full_name(self))

        # Save the object.
        self._save_impl(pickler)
        pickler.close()

    def _save_impl(self, pickler):
        """
        An function to implement save for the object in consideration.
        The default implementation will dump self to the pickler.

        WARNING: This implementation is very simple.
                 Overwrite for smarter implementations.

        Parameters
        ----------
        pickler : An opened GLPickle archive (Do not close the archive.)
        """
        if not self._is_gl_pickle_safe():
            raise RuntimeError("Cannot pickle object. Use the 'save' method.")

        # Pickle will hate the proxy
        self.__proxy__ = None
        pickler.dump(self)

    def summary(self, output=None):
        """
        Print a summary of the model. The summary includes a description of
        training data, options, hyper-parameters, and statistics measured
        during model creation.

        Parameters
        ----------
        output : str, None
            The type of summary to return.

            - None or 'stdout' : print directly to stdout.

            - 'str' : string of summary

            - 'dict' : a dict with 'sections' and 'section_titles' ordered
              lists. The entries in the 'sections' list are tuples of the form
              ('label', 'value').

        Examples
        --------
        >>> m.summary()
        """
        if output is None or output == 'stdout':
            pass
        elif (output == 'str'):
            return self.__repr__()
        elif output == 'dict':
            return _toolkit_serialize_summary_struct( self, \
                                            *self._get_summary_struct() )
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.summary')
        try:
            print(self.__repr__())
        except:
            return self.__class__.__name__

    def _get_wrapper(self):
        """
        Return a function: UnityModel -> M, for constructing model
        class M from a UnityModel proxy.

        Only used in GLC-1.3.

        """
        proxy_wrapper = self.__proxy__._get_wrapper()

        # Define the function
        def model_wrapper(unity_proxy):

            # Load the proxy object. This returns a proxy object with
            # 'temp_file' set to where the object is pickled.
            model_proxy = proxy_wrapper(unity_proxy)
            temp_file = model_proxy.temp_file

            # Setup the unpickler.
            unpickler = gl_pickle.GLUnpickler(temp_file)

            # Get the version
            version = unpickler.load()

            # Load the class name.
            cls_name = unpickler.load()
            cls = _get_class_from_name(cls_name)

            # Load the object with the right version.
            obj = cls._load_version(unpickler, version)

            # Return the object
            return obj

        return model_wrapper

    def _get_workflow(self):
        try:
            self.__workflow
        except AttributeError:
            self.__workflow = _ModelWorkflow()

        return self.__workflow

    def _get_version(self):
        return self._PYTHON_MODEL_VERSION

    def __getitem__(self, key):
        return self.get(key)

    def show(self, view=None, model_type='base'):
        """
        show(view=None)
        Visualize with GraphLab Canvas :mod:`~graphlab.canvas`. This function
        starts Canvas if it is not already running. If the model has already
        been plotted, this function will update the plot.

        Parameters
        ----------
        view : str, optional

            - 'Summary': The summary description of a Model.

            - 'Evaluation': A visual representation of the evaluation results
              for a Model.

        Returns
        -------
        view : graphlab.canvas.view.View
            An object representing the GraphLab Canvas view.

        See Also
        --------
        canvas

        Examples
        --------
        Suppose 'm' is a Model, we can view it in GraphLab Canvas using:

        >>> m.show()
        """
        from graphlab.visualization.show import show
        show(self, view=view, model_type=model_type)

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An function to load an object with a specific version of the class.

        WARNING: This implementation is very simple.
                 Overwrite for smarter implementations.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handle.

        version : int
            A version number as maintained by the class writer.
        """
        # Warning: This implementation is by default not version safe.
        # For version safe implementations, please write the logic
        # that is suitable for your model.
        return unpickler.load()

class PythonProxy(object):
    """
    Simple wrapper around a Python dict that exposes get/list_fields.
    This is used by ProxyBasedModel objects to hold internal state.
    """

    def __init__(self, state={}):
        self.state = _copy(state)
        
    def get(self, key):
        return self.state[key]

    def keys(self):
        return self.state.keys()

    def list_fields(self):
        """
        List of fields stored in the model. Each of these fields can be queried
        using the ``get(field)`` function or ``m[field]``.

        Returns
        -------
        out : list[str]
            A list of fields that can be queried using the ``get`` method.

        Examples
        --------
        >>> fields = m.list_fields()
        """
        return list(self.state.keys())

    def __contains__(self, key):
        return self.state.__contains__(key)

    def __getitem__(self, field):
        return self.state[field]

    def __setitem__(self, key, value):
        self.state[key] = value

    def __delitem__(self, key):
        del self.state[key]

    def pop(self, key):
        return self.state.pop(key)

    def update(self, d):
        self.state.update(d)

class ProxyBasedModel(object):
    """Mixin to use when a __proxy__ class attribute should be used for
    additional fields. This allows tab-complete (i.e., calling __dir__ on the
    object) to include class methods as well as the results of
    __proxy__.list_fields().
    """
    """The UnityModel Proxy Object"""
    __proxy__ = None

    def __dir__(self):
        """
        Combine the results of dir from the current class with the results of
        list_fields().
        """
        # Combine dir(current class), the proxy's fields, and the method
        # list_fields (which is hidden in __getattribute__'s implementation.
        return dir(self.__class__) + list(self.list_fields()) + ['list_fields']

    def get(self, field):
        """
        Return the value contained in the model's ``field``.

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out
            Value of the requested field.

        See Also
        --------
        list_fields
        """
        try:
            return self.__proxy__[field]
        except:
            raise ValueError("There is no model field called {}".format(field))

    def __getattribute__(self, attr):
        """
        Use the internal proxy object for obtaining list_fields.
        """
        proxy = object.__getattribute__(self, '__proxy__')

        # If no proxy exists, use the properties defined for the current class
        if proxy is None:
            return object.__getattribute__(self, attr)

        # Get the fields defined by the proxy object
        if not hasattr(proxy, 'list_fields'):
            fields = []
        else:
            fields = proxy.list_fields()

        def list_fields():
            """
            List of fields stored in the model. Each of these fields can be queried
            using the ``get(field)`` function or ``m[field]``.

            Returns
            -------
            out : list[str]
                A list of fields that can be queried using the ``get`` method.

            Examples
            --------
            >>> fields = m.list_fields()
            """
            return fields

        if attr == 'list_fields':
            return list_fields
        elif attr in fields:
            return self.get(attr)
        else:
            return object.__getattribute__(self, attr)

class Model(CustomModel, ProxyBasedModel):
    """
    This class defines the minimal interface of a model object
    storing the results from a toolkit function.

    User can query the model object via the `get` function, and
    the list of the queriable fields can be obtained via `list_fields`.

    Model object also supports dictionary type access, e.g. m['pagerank']
    is equivalent to m.get('pagerank').
    """


    def name(self):
        """
        Returns the name of the model.

        Returns
        -------
        out : str
            The name of the model object.

        Examples
        --------
        >>> model_name = m.name()
        """
        return self.__class__.__name__


    def get(self, field):
        """Return the value for the queried field.

        Each of these fields can be queried in one of two ways:

        >>> out = m['field']
        >>> out = m.get('field')  # equivalent to previous line

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out : value
            The current value of the requested field.

        """
        if field in self.list_fields():
            return self.__proxy__.get(field)
        else:
            raise KeyError('Field \"%s\" not in model. Available fields are'
                         '%s.') % (field, ', '.join(self.list_fields()))

    def __getitem__(self, key):
        return self.get(key)

    def _get_wrapper(self):
        """Return a lambda function: UnityModel -> M, for constructing model
        class M from a UnityModel proxy."""
        raise NotImplementedError

    def _get_ml_metric_config(self):
        '''Returns a metric configuration script used to define the type of metrics
        computed by the predictive service.'''
        return {}

    def save(self, location):
        """
        Save the model. The model is saved as a directory which can then be
        loaded using the :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        location : string
            Target destination for the model. Can be a local path or remote URL.

        See Also
        ----------
        graphlab.load_model

        Examples
        ----------
        >>> model.save('my_model_file')
        >>> loaded_model = graphlab.load_model('my_model_file')

        """
        _mt._get_metric_tracker().track('toolkit.model.save')
        return glconnect.get_unity().save_model(self, _make_internal_url(location))

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()

    @classmethod
    def _is_gl_pickle_safe(cls):
        """
        Return True if the model is GLPickle safe i.e if the model does not
        contain elements that are written using Python + GraphLab objects.
        """
        return False

class SDKModel(CustomModel, ProxyBasedModel):
    """
    This class defines the minimal interface of an SDK model object
    to be save-able and loadable from Python.
    """

    def __init__(self, model_proxy):
        self.__proxy__ = model_proxy

    def _get_wrapper(self):
        """Return a lambda function: UnityModel -> M, for constructing model
        class M from a UnityModel proxy.

        Example
        -------
        For example, for the demo class _demo_class(), this is the wrapper code
        that is neeeded to save and load.

        def _get_wrapper(self):
            def model_wrapper():
                return gl.extensions._demo_class()
            return model_wrapper
        """
        raise NotImplementedError

    def name(self):
        """
        Returns the name of the model.

        Returns
        -------
        out : str
            The name of the model object.

        Examples
        --------
        >>> model_name = m.name()
        """
        return self.__class__.__name__

    def get(self, field):
        """Return the value for the queried field.

        Each of these fields can be queried in one of two ways:

        >>> out = m['field']
        >>> out = m.get('field')  # equivalent to previous line

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out : value
            The current value of the requested field.

        """
        if field in self.list_fields():
            return self.__proxy__.get(field)
        else:
            raise KeyError('Field \"%s\" not in model. Available fields are'
                         '%s.') % (field, ', '.join(self.list_fields()))

    def __getitem__(self, key):
        return self.get(key)

    @classmethod
    def _is_gl_pickle_safe(cls):
        """
        Return True if the model is GLPickle safe i.e if the model does not
        contain elements that are written using Python + GraphLab objects.
        """
        return False

    def save(self, location):
        """
        Save the model. The model is saved as a directory which can then be
        loaded using the :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        location: str
            Target destination for the model. Can be a local path or remote URL.

        See Also
        ----------
        graphlab.load_model

        Examples
        --------
        >>> model.save('my_model_file')
        >>> loaded_model = gl.load_model('my_model_file')
        """

        return glconnect.get_unity().save_model(self.__proxy__,
                                                _make_internal_url(location),
                                                self._get_wrapper())
