import json as _json

from .. import _view

NoneType = type(None)

class PlotlyView(_view.View):
    def __init__(self, data, layout=None):
        super(PlotlyView, self).__init__(
            name='Plotly Wrapper View',
            label='Plot',
            tag_name='gl-plotly-wrapper',
            description='Plotly View inside the Turi Views platform'
        )
        self.__data = data
        self.__layout = layout

    def get_plot(self):
        import plotly
        figure = plotly.tools.return_figure_from_figure_or_data(self.__data, validate_figure=False)
        data = _json.dumps(figure.get('data', []), cls=plotly.utils.PlotlyJSONEncoder)
        if self.__layout is None:
            layout = _json.dumps(figure.get('layout', {}), cls=plotly.utils.PlotlyJSONEncoder)
        else:
            layout = self.__layout
        return {
            'data': data,
            'layout': layout
        }
