import graphlab.canvas.handlers
import graphlab.canvas.server
import graphlab.canvas.state
import graphlab.connect as _mt
from graphlab import version
from graphlab.canvas.utils import _to_json

import codecs
import os
import webbrowser

class Target(object):
    """
    A base class that serves as an interface for targets.
    """
    # instance variables
    state = None
    server = None

    def __init__(self, port=None):
        import graphlab.canvas.views.job_session
        import graphlab.canvas.views.data_objects
        import graphlab.canvas.views.model_objects
        from graphlab.deploy import jobs
        self.state = graphlab.canvas.state.State()
        # TODO server is not necessary in static IPython/Jupyter Notebook
        self.server = graphlab.canvas.server.Server(self.state, port)
        # add data objects to left nav
        DataView = graphlab.canvas.views.data_objects.DataObjectsView()
        self.add_variable(DataView.get_temporary_name(), DataView)
        # add models to left nav
        ModelObjectsView = graphlab.canvas.views.model_objects.ModelObjectsView()
        self.add_variable(ModelObjectsView.get_temporary_name(), ModelObjectsView)
        # add job dashboard to left nav
        jobsView = graphlab.canvas.views.job_session.ScopedSessionView(jobs)
        self.add_variable(jobsView.get_temporary_name(), jobsView)

    def get_asset_url(self):
        if version == "{{VERSION_STRING}}":
            return 'http://localhost:' + str(self.server.get_port()) + '/'
        else:
            return 'https://static.turi.com/products/graphlab-create/%s/canvas/' % version

    # methods
    def show(self, variable=None):
        # track metrics on variable type
        if variable is not None:
            _mt._get_metric_tracker().track('canvas.show.%s' % type(variable).__name__)
            self.state.set_selected_variable(variable)
        else:
            _mt._get_metric_tracker().track('canvas.show')
        self.server.start()

    def add_variable(self, name, variable):
        self.state.add_variable(name, variable)

class NoneTarget(Target):
    """
    Disable all Canvas output.
    """
    # override all parent methods to do nothing
    def show(self, variable=None):
        pass
    def add_variable(self, name, variable):
        pass

class HeadlessTarget(Target):
    """
    An interactive visualization canvas.
    """

    clientURL = None

    def show(self, variable=None):
        super(HeadlessTarget, self).show(variable)
        if not(self.server.alive()):
            self.clientURL = 'http://localhost:' + str(self.server.get_port()) + "/index.html"
            print('Canvas is accessible via web browser at the URL: %s' % self.clientURL)

class InteractiveTarget(HeadlessTarget):
    """
    An interactive browser-based visualization canvas. Opens a webbrowser locally.
    """
    def show(self, variable=None):
        super(InteractiveTarget, self).show(variable)
        if not(self.server.alive()):
            print('Opening Canvas in default web browser.')
            webbrowser.open(self.clientURL)
        else:
            print('Canvas is updated and available in a tab in the default browser.')

class IPythonTarget(Target):
    """
    Visualization rendered in an IPython Notebook or Jupyter Notebook output cell.
    """
    @staticmethod
    def __readFile(filename):
        """
        Read a file from the filesystem with utf-8 encoding and return the contents as a string.
        """
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, filename)
        contents = None
        with codecs.open(filename, encoding='utf-8') as f:
          contents = f.read()
        return contents

    def __getLibraries(self):
        """
        Generate a list of strings of all library dependencies.
        """
        libraries = [
            'ipython_app.js'
        ]

        return [self.get_asset_url() + 'js/' + js_file for js_file in libraries]

    def __makeJS(self, data, viewName, viewComponent):
        """
        Generate a JavaScript snippet that will load Canvas in an output cell
        of IPython Notebook or Jupyter Notebook.
        """
        # Code snippet to include dependency JS files, CSS styles, and a require call to run Canvas
        js = u"""
            (function(){

                var e = null;
                if (typeof element == 'undefined') {
                    var scripts = document.getElementsByTagName('script');
                    var thisScriptTag = scripts[scripts.length-1];
                    var parentDiv = thisScriptTag.parentNode;
                    e = document.createElement('div');
                    parentDiv.appendChild(e);
                } else {
                    e = element[0];
                }

                if (typeof requirejs !== 'undefined') {
                    // disable load timeout; ipython_app.js is large and can take a while to load.
                    requirejs.config({waitSeconds: 0});
                }

                require(%(dependencies)s, function(IPythonApp){
                    var app = new IPythonApp();
                    app.attachView('%(view)s','%(component)s', %(data)s, e);
                });
            })();
        """ % {
            "dependencies": self.__getLibraries(),
            "component": viewComponent,
            "view": viewName,
            "data": data
        }
        return js

    def show(self, variable=None):
        super(IPythonTarget, self).show(variable)
        selected = self.state.get_selected_variable()
        if selected is None:
            raise BaseException('No object is currently selected for viewing in Canvas. Call .show on a GraphLab Create object to show output in IPython Notebook or Jupyter Notebook.')
        (name, view) = selected
        assert(type(name) == tuple)
        assert(isinstance(view, graphlab.canvas.views.base.BaseView))

        data = None
        selected_variable = graphlab.canvas.handlers.expand_selected_variable(self.state)
        data = {
            'selected_variable': selected_variable,
            'ipython': True,
        }
        data.update(view.get_staticdata())

        import IPython
        IPython.core.display.display_javascript(
            IPython.core.display.Javascript(
                data=self.__makeJS(_to_json(data), view.get_js_file(), view.get_js_component()),
                css=['//cdnjs.cloudflare.com/ajax/libs/font-awesome/4.1.0/css/font-awesome.min.css', self.get_asset_url() + 'css/canvas.css']
            )
        )
