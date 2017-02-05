from graphlab.canvas.views.bivariate import BivariateView as _bivariate
import graphlab as gl
import six

class BoxWhiskerView(_bivariate):

    def __init__(self, obj, params):
        super(BoxWhiskerView, self).__init__(obj, params)
        self.col1 = self.get_column('x')
        self.col2 = self.get_numeric_column('y', self.col1)
        self.__re_init()

    def __re_init(self):
        self.__quantile_col1 = self.col1
        self.__quantile_col2 = self.col2
        if self.col1 is None or \
           self.col2 is None:
            return
        self.__quantile = gl.extensions._canvas.streaming.groupby.quantile()
        self.__quantile.init(gl.SFrame([self.obj[self.col1], self.obj[self.col2]]))

    def __get_values(self, materialize):
        if self.col1 != self.__quantile_col1 or \
           self.col2 != self.__quantile_col2:
            self.__re_init()

        if self.col1 is None or \
           self.col2 is None:
            return {
               'data': {},
               'progress': 1.0,
               'complete': 1
           }
        chunk = self.__quantile.get()
        if materialize:
            while not(self.__quantile.eof()):
                chunk = self.__quantile.get()
        # sort chunk grouped
        grouped_chunk = [ (k, v.tolist()) for (k,v) in six.iteritems(chunk.grouped)]
        grouped_chunk.sort(key=lambda x: x[0])
        return {
            'data': {
                'grouped': grouped_chunk,
                'omitted': chunk.omitted
            },
            'progress': float(self.__quantile.rows_processed) / len(self.obj),
            'complete': self.__quantile.eof()
        }

    def _get_values(self, url, handler):
        handler.write(self.__get_values(False))

    def get_staticdata(self):
        return self.__get_values(True)
