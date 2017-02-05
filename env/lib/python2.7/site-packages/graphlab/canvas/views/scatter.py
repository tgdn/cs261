from graphlab.canvas.views.bivariate import BivariateView as _bivariate
import graphlab as gl
import datetime

import sys
if sys.version_info.major == 3:
    long = int


class ScatterPlot(_bivariate):

    def __init__(self, obj, params):
        # x is the name of the column in the X axis
        # y is the name of the column in the Y axis
        super(ScatterPlot, self).__init__(obj, params)
        self.col1 = self.get_typed_column('x', types = [int,float,long,datetime.datetime])
        self.col2 = self.get_numeric_column('y', avoid = self.col1)
        self.__re_init()

    def __re_init(self):
        # TODO this is a hack -- we should better keep track of which columns
        # are currently being used.
        self.__scatter_col1 = self.col1
        self.__scatter_col2 = self.col2
        if self.col1 is None or \
           self.col2 is None:
            return

        self.__scatter = gl.extensions._canvas.streaming.sframe_sample()
        self.__scatter.init(gl.SFrame([self.obj[self.col1], self.obj[self.col2]]))

    def __get_values(self, materialize):
        #making the array of dictionary of points
        if self.col1 != self.__scatter_col1 or \
           self.col2 != self.__scatter_col2:
            self.__re_init()

        if self.col1 is None or \
           self.col2 is None:
           return {
               'data': [],
               'progress': 1.0,
               'complete': 1
           }

        sample = self.__scatter.get()
        if materialize:
            while not(self.__scatter.eof()):
                sample = self.__scatter.get()

        # drop na values
        sample = [x for x in sample if None not in x]

        return {
            'data': sample,
            'progress': float(self.__scatter.rows_processed) / len(self.obj),
            'complete': self.__scatter.eof()
        }

    def _get_values(self, url, handler):
        handler.write(self.__get_values(False))

    def get_staticdata(self):
        return self.__get_values(True)
