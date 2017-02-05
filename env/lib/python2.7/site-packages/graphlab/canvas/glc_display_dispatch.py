from graphlab.visualization import show, show_dispatch

from graphlab.data_structures.sframe import SFrame
@show_dispatch(SFrame)
def show(obj, **kwargs):
    import graphlab.canvas
    import graphlab.canvas.inspect
    import graphlab.canvas.views.sframe
    graphlab.canvas.inspect.find_vars(obj)
    return graphlab.canvas.show(graphlab.canvas.views.sframe.SFrameView(obj, params=kwargs))


from graphlab.data_structures.sarray import SArray
@show_dispatch(SArray)
def show(obj, **kwargs):
    import graphlab.canvas
    import graphlab.canvas.inspect
    import graphlab.canvas.views.sarray

    graphlab.canvas.inspect.find_vars(obj)
    return graphlab.canvas.show(graphlab.canvas.views.sarray.SArrayView(obj, params=kwargs))



from graphlab.data_structures.sgraph import SGraph
@show_dispatch(SGraph)
def show(obj, **kwargs):
    import graphlab.connect as _mt
    _mt._get_metric_tracker().track('sgraph.show')

    import graphlab.canvas
    import graphlab.canvas.inspect
    import graphlab.canvas.views.sgraph

    graphlab.canvas.inspect.find_vars(obj)
    if 'highlight' in kwargs:
        highlight = kwargs['highlight']
        if isinstance(highlight, SArray):
            # convert to list
            highlight = list(highlight)
            kwargs['highlight'] = highlight
        if isinstance(highlight, list):
            # convert to dict
            highlight_color = kwargs['highlight_color'] if 'highlight_color' in kwargs else []
            highlight_color = [highlight_color] * len(highlight)
            highlight = dict(zip(highlight, highlight_color))
            kwargs['highlight'] = highlight
            kwargs['highlight_color'] = highlight_color
    return graphlab.canvas.show(graphlab.canvas.views.sgraph.SGraphView(obj, params=kwargs))



from graphlab.toolkits._model import Model, CustomModel
@show_dispatch((Model, CustomModel))
def show(obj, **kwargs):
    import graphlab.canvas.inspect
    import graphlab.canvas.views.model
    graphlab.canvas.inspect.find_vars(obj)
    return graphlab.canvas.show(graphlab.canvas.views.model.ModelView(obj, params=kwargs))
