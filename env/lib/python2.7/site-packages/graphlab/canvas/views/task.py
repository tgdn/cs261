import graphlab.canvas.views.base


class TaskView(graphlab.canvas.views.base.BaseView):

    def __init__(self, obj, params=None):
        super(TaskView, self).__init__(obj, params)
        self.cached_sketch = None

    def get_metadata(self):
        return {
            'taskname' : [self.obj.get_name()],
            'description' : self.obj.get_description(),
            'inputs' : self.obj.get_inputs(),
            'outputs' : self.obj.get_outputs(),
            'code' : self.obj.get_code(),
            'params' : self.obj.get_params(),
            'required_pkg' : list(self.obj.get_required_packages()),
        }

    def get_staticdata(self):
        return self.get_metadata()

    def get_js_file(self):
        return 'task'

