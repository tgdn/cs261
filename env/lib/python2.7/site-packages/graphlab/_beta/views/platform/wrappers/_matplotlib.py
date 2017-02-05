try:
    import io as _StringIO
except ImportError:
    import StringIO as _StringIO

from .. import _view
from ._plotly import PlotlyView as _PlotlyView

class MatplotlibView(_view.View):
    def __init__(self, figure, backend='plotly'):
        """
        Wraps a matplotlib figure in a View. The resulting View can be
        used like any other view (i.e. shown, composed in a layout, etc.).

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            A matplotlib figure.

        backend : str
            Can be one of the following values: 'plotly' or 'svg'. The default
            is 'plotly'. If 'plotly' is selected, the plotly wrapper for
            matplotlib is used (in combination with
            `graphlab.beta.views.wrappers.PlotlyView`). The plotly wrapper
            allows for more user interaction with the plot but has some
            limitations and incompatibilities. If 'svg' is selected, the plot
            will be serialized to non-interactive SVG format and that
            representation will be displayed in the view.

        Returns
        -------
        out : View
            A View representation of the figure.
        """
        super(MatplotlibView, self).__init__(
            name='Matplotlib Wrapper View',
            label='Plot',
            tag_name='gl-matplotlib-wrapper',
            description='Matplotlib Plot inside the Turi Views platform'
        )
        self.__backend = backend
        if backend == 'plotly':
            import plotly
            self.__plotly_view = _PlotlyView(plotly.tools.mpl_to_plotly(figure))
        elif backend == 'svg':
            buf = _StringIO.StringIO()
            figure.savefig(buf, format='svg')
            self.__svg = buf.getvalue()
        else:
            raise ValueError('Expected backend to be one of: "plotly", "svg"')

    def get_plot(self):
        backend = self.__backend
        plot = None
        if backend == 'plotly':
            plot = self.__plotly_view.get_plot()
        else:
            assert backend == 'svg'
            plot = self.__svg
        return {
            "backend": backend,
            "plot": plot
        }
