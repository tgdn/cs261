import graphlab.canvas
import graphlab.canvas.views.base

class ModelView(graphlab.canvas.views.base.BaseView):

    def __init__(self, obj, params=None):
        super(ModelView, self).__init__(obj, params)

        if not('model_type' in self.params) or self.params['model_type'] is None:
            self.params['model_type'] = 'base'

        if not('view' in self.params) or self.params['view'] is None:
            self.params['view'] = 'Summary'

        self.objectType = 'Model'
        # handlers for Model View
        self.register_handler('get', 'model', self.__get_attributes)



    def __get_attributes(self, url, handler):

        attributes = self.__get_model_attributes()

        handler.write({
            'model_type': self.params['model_type'],
            'attributes': attributes
        })

    def __downsample_roc_curve(self, roc_curve):
        if len(roc_curve) == 1001:
            return roc_curve
        assert len(roc_curve) == 100001, 'Expected roc_curve to have 1e5+1 data points'
        downsampled = roc_curve[::100]
        assert len(downsampled) == 1001, 'Expected downsampled roc_curve to have 1e3+1 data points'
        return downsampled

    def __munge_workflows(self, step):
        if not('roc_curve' in step):
            return step
        if 'class' in step['roc_curve'].column_names():
            del step['roc_curve']
        else:
            # downsample roc_curve to 1,000 rows
            step['roc_curve'] = self.__downsample_roc_curve(step['roc_curve'])
        return step

    def __get_workflows(self):
        #workflow_steps is a list of tuples of (step_label, step_results/sframe)
        #cast to dict before sending out to response
        # label name of the dataset
        # graphlab.canvas.views.base._unbox_data_structure convert dict sframe to JOSN compitable object
        return [ [label, graphlab.canvas.views.base._unbox_data_structure(self.__munge_workflows(step))] \
                for (label, step) in self.obj._get_workflow().get_steps() ]

    def __get_comparison(self):
        # send the packed comparison result to the client side;
        return graphlab.canvas.views.base._unbox_data_structure(self.obj._get_workflow().get_comparison())

    def get_metadata(self):
            return {
                'evaluations': self.__get_workflows(),
                'comparison': self.__get_comparison(),
                'attributes': self.__get_model_attributes(),
                'model_type': self.params['model_type'],
                'view_params': self.params
            }

    def __get_model_attributes(self):
        # get the model summaries: parameters training
        attributes = self.obj.summary(output='dict')
        return attributes

    def get_js_file(self):
        return 'model'

    def get_js_components(self):
        return [
            'Summary',
            'Evaluation',
            'Comparison'
        ]

    def get_js_component(self):
        if 'view' in self.params:
            name = self.params['view']
            if name is None:
                return 'Summary'
            if name == 'Summary' or \
                name == 'Evaluation' or \
                name == 'Comparison':
                return name # go to top-level tabs
            import logging
            logging.warning('view param %s is not recognized. It has to be either "Summary", "Evaluation" or "Comparison"' % name)
        #default to Summary view
        return 'Summary'

    @classmethod
    def __retrieve_nested_key(cls, dataDict, keyList):
        try:
            return reduce(lambda d, k: d[k], keyList, dataDict)
        except KeyError:
            pass
        return None
