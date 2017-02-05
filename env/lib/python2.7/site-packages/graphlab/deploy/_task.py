# Other imports
import copy as _copy
import logging as _logging
import inspect as _inspect
import sys as _sys

# Graphlab imports
from . import _internal_utils
from graphlab.util import _raise_error_if_not_function, _raise_error_if_not_of_type
from ._artifact import Artifact as _Artifact


__LOGGER__ = _logging.getLogger(__name__)

# We keep track of task version.
# -----------------------------------------------------------------------------
#

class Task(_Artifact):
    """
    A Task encapsulates a block of code.

    A Task is a modular, composable, reusable component that encapsulates a
    unit-of-work. It is the robust building block for building GraphLab Create
    Data Pipelines.

    Tasks allow for code to logically reference inputs, and output
    by name, so the code is reusable in many places.

    Tasks make it easy to build a data pipeline of composable parts, each
    having a single responsibility in the pipeline, with the tasks loosely
    coupled. All inputs, and output can be bound at runtime.

    Parameters
    ----------
    func: func
        Name of the function from which the task is created.

    name : str
        Name of the Task, this is how the Task will be retrieved in subsequent
        retrievals.

    description : str, optional
        Description for this Task, optional.

    Returns
    -------
    out : :class:`~graphlab.deploy._task.Task`
        A new Task.

    Notes
    -----
    - Task names must be unique.

    Examples
    --------
    An example Task definition, for generating random names and building an
    :class:`~graphlab.SFrame` from them:

    >>> def name_task(number = 10):
           import names
           people = [names.get_full_name() for i in range(number)]
           sf = graphlab.SFrame({'names':people})
           return sf

    >>> names = graphlab.deploy._task.Task(name_task, 'names')
    """

    # Static state
    _typename = 'Task'
    _data = {}

    #
    #  Version 1: GLC 1.3
    #  --------------------
    #  Everything starts from scratch here. Everything before V1.3 is not version
    #  aware and will not be backwards compatible.
    #
    _TASK_VERSION = 1

    def __init__(self, func, name=None, description=None):
        """
        Create a new Task specifying its name and optionally a description.
        """

        # Must be a function
        _raise_error_if_not_function(func, "func")

        # Set the name
        name = func.__name__ if not name else name
        _raise_error_if_not_of_type(name, str, "name")

        self.name = name
        self._data = dict()
        self._data['code'] = None
        self._data['codestr'] = None
        self._data['inputs'] = dict()
        self._data['output'] = None
        self._data['packages'] = set()
        self._data['description'] = ''
        self._modified_since_last_saved = None

        if description is not None:
            self.set_description(description)

        # Inspect the function.
        specs = _inspect.getargspec(func)
        varargs = specs.varargs
        defaults = _copy.copy(specs.defaults)
        args = _copy.copy(specs.args)

        # Set the code to function arguments + *args + **kwargs
        self.set_code(func)

        # Set the inputs
        all_args = _copy.copy(args)
        if varargs:
            all_args.append(varargs)
        self.set_inputs(all_args)

        # Bind default values
        if defaults:
            for index, arg in enumerate(args[-len(defaults):]):
                self.set_inputs({arg : defaults[index]})

        # Set required packages
        if _sys.version_info.major == 3:
            func_dict = func.__dict__
        else:
            func_dict = func.func_dict

    def __hash__(self):
        return id(self) // 16

    def set_name(self, name):
        """
        Set the name of the Task, which must be unique.

        Parameters
        ----------
        name : str
            Name of the Task.

        Returns
        -------
        self : Task
        """

        _raise_error_if_not_of_type(name, str, "name")
        self.name = str(name)
        self._set_dirty_bit()
        return self

    def set_description(self, description):
        """
        Set the description for this Task.

        Parameters
        ----------
        description : str
            A description for the Task.

        Returns
        -------
        self : Task
        """
        _raise_error_if_not_of_type(description, str, "description")
        self._data['description'] = description
        self._set_dirty_bit()
        return self

    def clone(self, name):
        """
        Make a copy of the current Task with a new name. The new task will have
        a different name, but will share all other task metadata like inputs,
        output, description, and code.

        Parameters
        ----------
        name : str
            Name of the cloned Task.

        Returns
        -------
        out
            A newly cloned Task instance.
        """
        new = Task.__new__(Task)
        new.name = name

        # Shallow copy
        new._data = _copy.copy(self._data)

        # Shallow copy of (individual objects that are accesed by reference)
        for key in ['inputs', 'output', 'packages']:
            new._data[key] = _copy.copy(self._data[key])

        new._modified_since_last_saved = self._modified_since_last_saved
        return new

    def set_inputs_from_task(self, names):
        """
        Set input(s) for this Task from the output of another task.

        Inputs can be any object that can be pickled using GL-Pickle but must
        come from the output of another task.

        Parameters
        ----------
        names : dict (str, Task)
            Each key in the dict is considered a name for an input in this
            Task. Values for the dict must be a Task which then binds the output
            of the Task to a particular input.

        Returns
        -------
        self : Task

        See Also
        --------
        set_output

        Examples
        --------

        >>> t3 = graphlab.deploy._task.Task(my_func, 'set_inputs_ex3')
        >>> t3.set_inputs_from_task({'d' : t2})

        """
        if names is None:
            raise TypeError('Names are required while binding two tasks.')

        if isinstance(names, dict):
            for key, from_task in names.items():
                self._set_one_input(name=key, from_task=from_task, delete=False)

        return self

    def set_inputs(self, names):
        """
        Set input(s) for this Task.

        Inputs can be any object that can be pickled using GL-Pickle but cannot
        come from the output of another task. For that, use the
        set_inputs_from_task function.

        Parameters
        ----------
        names : list [str] | dict [str, obj]
            If a dict is provided, then each key is considered a name for an
            input in this Task, and each value is considered the definition of the
            input.

            When a list is provided,  then each entry is considered a name for
            an input in this Task, and the value for that slot is set to None.

        Returns
        -------
        self : Task

        See Also
        --------
        set_output

        Examples
        --------
        To define only input names for a task, use a list of strings:


        >>> # For late binding
        >>> t1 = graphlab.deploy._task.Task(my_func, 'set_inputs_ex1')
        >>> t1.set_inputs(['one', 'two', 'three'])

        >>> # For early binding
        >>> t3 = graphlab.deploy._task.Task(my_func, 'set_inputs_ex3')
        >>> t3.set_inputs({
        ...     'b' : 'set_inputs_ex2',
        ...     'c' : 'foo',
        ...     'd' : ('foo', 'bar')})
        """
        if names is None:
            raise TypeError('Names are required while binding two tasks.')
        _raise_error_if_not_of_type(names, [list, dict], 'names')

        if isinstance(names, list):
            for name in set(names):
                self._set_one_input(name=name, delete=False)
        elif isinstance(names, dict):
            for key, value in names.items():
                self._set_one_input(name=key, value=value, delete=False)
        return self

    def delete_inputs(self, names):
        """
        Set input(s) for this Task.

        Inputs can be any object that can be pickled using GL-Pickle but cannot
        come from the output of another task. For that, use the
        set_inputs_from_task function.

        Parameters
        ----------
        names : list [str]

            When a list is provided,  then each entry is considered a name for
            an input in this Task, and is hence removed.

        Returns
        -------
        self : Task

        See Also
        --------
        delete_output

        Examples
        --------
        To define only input names for a task, use a list of strings:


        >>> # For late binding
        >>> t1 = graphlab.deploy._task.Task(my_func, 'set_inputs_ex1')
        >>> t1.delete_inputs(['one', 'two', 'three'])

        """
        if names is None:
            return self
        _raise_error_if_not_of_type(names, [list], 'names')

        for name in set(names):
            self._set_one_input(name=name, delete=True)

        return self

    def _set_one_input(self, name='input', value=None, from_task=None, delete=False):
        """
        Set/Update an input for this Task.

        Parameters
        ----------

        name : str
            Name for this input. This will be how the code refers to this
            input at runtime. Default is 'input'.

        value : obj (supported by GL Pickle)
            Value for the object refered to using 'name'.

        from_task : Task|str

            Dependent Task to set as input, specifying the tuple with: (Task,
            output_name). Tasks can be referred to either by name or by
            reference. The output_name needs to be a string.

            For example, if the following is specified:

            >>> task._set_one_input(name='in', from_task='dep')

            then an input named 'in' will be defined on this Task, which
            has a dependency on the output of the Task named 'dep'.


        delete : bool, optional
            If delete is set to True then the name input is removed.
        """

        _raise_error_if_not_of_type(name, str, "name")
        _raise_error_if_not_of_type(from_task,
                       [type(None), Task], "from_task")

        # Delete the input.
        if delete is True and name in self._data['inputs']:
            del self._data['inputs'][name]
            return self

        # Early binding: Set the input
        if from_task is None:
            self._data['inputs'][name] = value
            self._set_dirty_bit()
            return self

        # Late binding: Set an input from a task.
        elif isinstance(from_task, Task):
            task = from_task
            self._data['inputs'][name] = task
            self._set_dirty_bit()
            return self

    def set_code(self, code):
        """
        Set the code block to run when Task is executed.

        The code to be run needs to be a function that takes one argument. When
        this function is called, the arguments will be the inputs and the return
        will be in the output.

        The inputs dictionary will have instantiated data sources by name. The
        output dictionary needs to be assigned by name to the results to save.

        Parameters
        ----------
        code : function
            Function to be called when this Task is executed.

        Returns
        -------
        self : Task

        Examples
        --------
        Using a defined function:

        >>> def func(task):
        >>>     input = task.inputs['input']
        >>>     task.output['output'] = input.apply(lambda x : x * 2)

        >>> t1 = graphlab.deploy._task.Task("set_code_ex1")
        >>> t1.set_code(func)

        """

        # Make sure it is a function.
        _raise_error_if_not_function(code)

        # Cannot work with instance method
        if(_inspect.ismethod(code)):
            raise TypeError(("Function cannot be an instance method, please"
                       " use a function."))

        # code is callable, so store it as is
        self._data['code'] = code
        self._data['codestr'] = _inspect.getsource(code)
        self._set_dirty_bit()


    def set_required_packages(self, packages=None):
        """
        Set the required packages for running this Task. When running this
        Task in a remote setting/environment, these packages are installed
        prior to execution.

        Parameters
        ----------
        packages : list | set [str]
            List of package requirements (same as disutils.requires) format for
            packages required for running this Task.

        Returns
        -------
        self : Task

        Notes
        -----
        - No validation is performed on the specified packages list. So for
          example, if a required package is specified twice with two different
          versions, this API will not complain.

        Examples
        --------
        >>> t = graphlab.deploy._task.Task('set_required_packages_ex1')
        >>> t.set_required_packages(['numpy==1.8.0', 'pandas==0.13.1'])
        """

        self._data['packages'].update(packages)
        self._set_dirty_bit()
        return self

    def _save_impl(self, pickler):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : An opened gl_pickle archive (DO NOT CLOSE after completion)
        """
        pickler.dump(self)

    def _get_version(self):
        return self._TASK_VERSION

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : A GLUnpickler archive.
        """
        # Load the dump.
        obj = unpickler.load()

        # Construct a new object.
        _data = obj._data
        new = cls(_data['code'], obj.name, _data['description'])
        assert obj._get_version() <= new._get_version()

        # Now copy over the useful parts of the dump.
        lst = ['_data', '_modified_since_last_saved', '_typename', 'name']
        _internal_utils.copy_attributes(new, obj, lst)
        return new

    def get_name(self):
        """
        Return the name of this Task.
        """
        return self.name

    def get_description(self):
        """
        Return the description of this Task.
        """
        return str(self._data['description'])

    def get_inputs(self):
        """
        Return a copy of the dictionary of inputs, by name.
        """
        return _copy.copy(self._data['inputs'])

    def get_code(self):
        """
        Return a human-readable string indicating the code that is set for
        this Task.
        """
        return self.__code_str__()

    def get_runnable_code(self):
        """
        Return machine-callable version of the code stored in Task. Not
        intended for interactive usage.
        """
        return self._data['code']

    def get_required_packages(self):
        """
        Return a copy of the set of packages (in distutils.requires format) required for
        this Task.

        References
        ----------
        - `Python distutils <https://docs.python.org/2/library/distutils.html>`_
        """
        return _copy.copy(self._data['packages'])

    def get_output(self):
        """
        Return a copy of the output dictionary for the named output for this Task.
        """
        return _copy.copy(self._data['output'])


    def run(self, *args, **kwargs):
        """
        Run this Task standalone for development/debugging purposes.

        Parameters
        ----------
        args   : tuple
            A tuple of ordered arguments.

        kwrags : dict
            A dictionary of keyword arguments.

        Returns
        -------

        func(*args, **kwargs) where func is the function encapsulated in the
        task.

        Examples
        --------
        >>> results = t2.run(1, 2, foo = bar)

        """
        func = self.get_runnable_code()
        if func == None:
            raise ValueError("No code to run, task not fully initialized.")

        return func(*args, **kwargs)

    def __code_str__(self):
        code = self._data['code']
        if code is None:
            return ""
        if 'codestr' in self._data:
            return "%s" % self._data['codestr']

    def __inputs_str__(self):
        inputs = self.get_inputs()
        ret = {}
        for key, value in inputs.items():
            if isinstance(value, Task):
                value = "(output from task: %s)" % (value.get_name())
            ret[key] = value
        return str(ret)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        header =   "Task        : %s" % (self.get_name())
        desc   =   "Description : %s" % (self.get_description())
        inputs =   "Input(s)    : %s" % self.__inputs_str__()
        output =   "Output      : %s" % self.get_output()
        packages = "Package(s)  : %s" % list(self.get_required_packages())
        code    =  "Code        :\n\n%s" % self.__code_str__()

        out = "\n".join([header, desc, inputs, output, packages, code])

        if hasattr(self, 'status'):
            out += "Status      : %s\n" % self.status
        return out
