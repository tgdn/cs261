from functools import wraps
import graphlab
import array as _array

temporarySFrameString = '<SFrame>'
temporarySArrayString = '<SArray>'

def _collect_model_workflow(f):
    """
    Decorator function to annotate and add a model evaluation result to a model.
    Canvas server will send the _ModelWorkflow list of results collected by this decorator.
    Only use this decorator on model methods.
    """
    @wraps(f)
    def wrapper(model, *args, **kwargs):
        result = f(model, *args, **kwargs)
        dataset_label = None
        dataset = []
        # model inherits from CustomModel
        if isinstance(model, graphlab.toolkits._model.CustomModel):
            if len(args) >= 1:
                dataset = args[0]
            elif 'dataset' in kwargs:
                dataset = kwargs['dataset']
            elif 'train_data' in kwargs:
                dataset = kwargs['train_data']

            if isinstance(dataset, ( graphlab.data_structures.sframe.SFrame,
                                     graphlab.data_structures.sarray.SArray,
                                     dict,
                                     list,
                                     tuple, _array.array) ):
                from graphlab.canvas.inspect import _find_variable_name
                (dataset_label, dataset_variable) = _find_variable_name(dataset)
                 
            if dataset_label is None:
                if isinstance(dataset_variable, graphlab.data_structures.sframe.SFrame):
                    dataset_label = temporarySFrameString
                elif isinstance(dataset_variable, graphlab.data_structures.sarray.SArray):
                    dataset_label = temporarySArrayString
                elif isinstance(dataset_variable, (dict, list, tuple, _array.array) ):
                    dataset_label = "<%s>" % str(type(dataset_variable))

            model._get_workflow().add_step(dataset_label, result)
        return result
    return wrapper


class _ModelWorkflow(object):
    """
    A collection class which is used by models to record workflow operations
    (e.g. evaluate)
    """

    def __init__(self):
        self.__steps = []
        self.__eval_results = None

    def add_step(self, step_type, step_result):
        """
        Add a workflow operation by label

        Parameters
        ----------
        step_type : string
            String label describing the operation

        step_result : SFrame/SArray
            The result returned by the operation
        """

        # __steps stores one item
        if len(self.__steps) >= 1:
            del self.__steps[:]
        self.__steps.insert(0, (step_type, step_result) )

    def get_steps(self):
        """
        Get the list of operations performed for a given model

        Return
        ----------
        steps : list of tuples
            each tuple is (operation_label, result_sframe)
        """
        return self.__steps

    def add_comparison(self, comp_obj):
        self.__eval_results = comp_obj

    def get_comparison(self):
        return self.__eval_results
