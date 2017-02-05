import graphlab.canvas.views.base

class ModelObjectsView(graphlab.canvas.views.base.BaseView):

    def __init__(self):
        super(ModelObjectsView, self).__init__(None, None)

    def get_js_file(self):
        return 'model_objects'

    def get_js_component(self):
        return 'View'

    def get_temporary_name(self):
        return ('Models',)
