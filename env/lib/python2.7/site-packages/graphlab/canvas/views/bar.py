from graphlab.canvas.views.bivariate import BivariateView as _bivariate
import graphlab as gl
import six

class BarGraphView(_bivariate):

    def __init__(self, obj, params):
        super(BarGraphView, self).__init__(obj, params)
        self.col1 = self.get_column('x')
        self.col2 = self.get_numeric_column('y', self.col1)
        self.__re_init()

    def __re_init(self):
        self.__summary_col1 = self.col1
        self.__summary_col2 = self.col2
        if self.col1 is None or \
           self.col2 is None:
            return
        self.__summary = gl.extensions._canvas.streaming.groupby.summary()
        self.__summary.init(gl.SFrame([self.obj[self.col1], self.obj[self.col2]]))

    def __get_values(self, materialize):
        if self.col1 != self.__summary_col1 or \
           self.col2 != self.__summary_col2:
            self.__re_init()

        if self.col1 is None or \
           self.col2 is None:
            return {
               'data': {},
               'progress': 1.0,
               'complete': 1
           }
        chunk = self.__summary.get()
        if materialize:
            while not(self.__summary.eof()):
                chunk = self.__summary.get()
        grouped_chunk = [ (k, v) for (k,v) in six.iteritems(chunk.grouped)]
        grouped_chunk.sort(key=lambda x: x[0])
        return {
            'data': {
                'grouped': grouped_chunk,
                'omitted': chunk.omitted
            },
            'progress': float(self.__summary.rows_processed) / len(self.obj),
            'complete': self.__summary.eof()
        }

    def _get_values(self, url, handler):
        handler.write(self.__get_values(False))

    def get_staticdata(self):
        return self.__get_values(True)
