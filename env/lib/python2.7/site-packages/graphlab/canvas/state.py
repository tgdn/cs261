"""
The state within this module represents the existence and selection state of
Python objects in the GraphLab canvas UI. The graphlab.canvas.server module
uses this wrapper to look up registered variables and selected
variables/columns. Access to the state in this module is intended to be
thread-safe.

Selection state is modeled on variables and columns. The Canvas UI presents
a list of variables known to it, single-selectable. When a variable is
selected, the Canvas UI presents a list of columns in that variable, also
single-selectable.
"""

import graphlab
import graphlab.canvas
import graphlab.canvas.views.base
import graphlab.connect as _mt
import graphlab.data_structures.gframe

from collections import OrderedDict
import threading
import traceback

class State:
    #
    # private variables
    #

    # instance variables
    # "name" for vars is always a tuple (fully qualified name, as in breadcrumb navigation)
    __vars = OrderedDict() # dictionary of name->view for GraphLab Create data structures discovered from stack frames
    __var_ids = {} # dictionary of id (long)->name (tuple) and name->id for GraphLab Create data structures in Canvas (two way map)
    __selected_var = None # currently selected variable reference
    __var_lock = threading.RLock() # use reentrant lock so methods here can easily call each other but all make sure we have a lock
    __exceptions = [] # exceptions to display in the UI (exceptions caught from handlers)
    __var_count = 0 # only used for metric tracking (previous count of variables so we don't over-count in metrics)

    #
    # public API
    #

    def get_variables(self):
        """
        Gets a list of Python variable names exposed to visualization.

        Returns
        -------
        vars : list[str]
            A list of variable names exposed to Canvas UI.
        """
        with self.__var_lock:
            names = list(filter(lambda n: len(n) == 1, reversed(list(self.__vars.keys()))))
            return [{
              'name': name,
              'type': self.__vars[name].objectType
            } for name in names]

    def add_variable(self, name, ref):
        """
        Add a variable reference by name.

        Parameters
        ----------
        name : str | tuple
            The name of a variable to store.
            If it is a tuple, it gives a contextual name like ('sf', 'col').
            Such a contextual name represents 'col' (SArray) as a child of 'sf'
            (SFrame).

        ref : *
            A reference to the variable to store or an already-instantiated View
            object encapsulating that variable (ex. SFrame, or SFrameView).
        """
        assert(type(name) == tuple)
        view = None
        if isinstance(ref, graphlab.canvas.views.base.BaseView):
            # use the reference and view passed in
            view = ref
            ref = ref.obj
        with self.__var_lock:
            # add the variable
            if view is not None:
                self.__vars[name] = view
            obj_id = graphlab.canvas._get_id(ref)
            self.__var_ids[name] = obj_id
            self.__var_ids[obj_id] = name

            # new variable to canvas, track it
            self.__track_variable_count()

    def __lookup_var(self, name):
        assert(type(name) == tuple)
        name = list(name)
        current = self.__vars[(name[0],)]
        rest = name[1:]
        while len(rest) > 0:
            current = current.child_views()[rest[0]]
            rest = rest[1:]
        return current

    def set_selected_variable(self, var):
        """
        Marks the variable passed in by name as selected in the UI.

        Parameters
        ----------
        var : str | unicode | tuple | SFrame | SArray
                The variable to select (by name or reference).
        """
        name = None
        if isinstance(var, tuple):
            # look up by name
            name = var
            var = self.__lookup_var(name)
        else:
            ref = var
            if isinstance(var, graphlab.canvas.views.base.BaseView):
                # use underlying object, not view wrapper
                ref = var.obj

            name = self.__find_name(ref)
            if name is None:
                # if we can't find the name, add it as anonymous
                name = var.get_temporary_name()
            # make sure this variable exists.
            self.add_variable(name, var)
        
        # tracks type of variable added to Canvas
        _mt._get_metric_tracker().track('canvas.set_selected_variable.%s' % type(var).__name__)

        with self.__var_lock:
            self.__selected_var = (name, self.__lookup_var(name))

    def __count_top_level_objects(self):
        # Returns the total number of "top level" (shown in left nav) Canvas objects.
        with self.__var_lock:
            previous = self.__var_count
            self.__var_count = len(list(filter(lambda x: len(x) == 1, self.__vars.keys())))
            return previous != self.__var_count

    def __track_variable_count(self):
        # track total count of variables in Canvas
        # we don't know how many the user will end up with in total so we'll count each increase.
        # for reporting we can estimate an aggregate "total" by subtracting from each count
        # (someone who had 5 variables in Canvas also at one point had 4, 3, 2, and 1, etc.)
        if (self.__count_top_level_objects()):
            _mt._get_metric_tracker().track('canvas.variable_count.%s' % self.__var_count)

    def __find_name(self, var):
        # return a name if we "know" about a variable
        # (recursively searching SArray/SFrame structure)
        # or None if we do not find it
        if var is None:
            return None
        obj_id = graphlab.canvas._get_id(var)
        with self.__var_lock:
            if obj_id in self.__var_ids:
                return self.__var_ids[obj_id]
        return None

    def get_selected_variable(self):
        """
        Gets a reference to the currently selected variable.

        Returns
        -------
        v : (str, (SFrame | SArray)) | None
            If there is a selected variable, returns a tuple of (name, reference) to that variable. Otherwise returns None.
        """
        with self.__var_lock:
            return self.__selected_var

    def add_exception(self, e):
        """
        Add a Python exception to show in the UI.

        Parameters
        ----------
        e : (type, value, traceback)
            A tuple of exception values as returned by sys.exc_info.
        """
        # don't print errors if the unity_server process is no longer running
        # this is normal on Python process shutdown
        try:
            if (graphlab.connect.main.get_server().proc):
                print('[ERROR] GraphLab Canvas: %s' % str(e))
        except:
            pass

        # truncate type, message, stack_trace to a reasonable length
        # (100 for type/message, 1000 for stack_trace)
        # so that we don't hit any size limits on librato or mixpanel
        properties = {
            'type': e[0].__name__[:100],
            'message': str(e[1])[:100],
            'stack_trace': traceback.format_tb(e[2])[:1000]
        }
        _mt._get_metric_tracker().track('canvas.unhandled_exception', properties=properties)
        with self.__var_lock:
            self.__exceptions.append(properties)

    def get_exceptions(self):
        """
        Retrive a list of all exceptions since the last call to get_exceptions

        Returns
        -------
        exceptions : [(type, value, traceback)]
        """
        exceptions = None
        with self.__var_lock:
            exceptions = self.__exceptions[:]
            self.__exceptions = []
        return exceptions

