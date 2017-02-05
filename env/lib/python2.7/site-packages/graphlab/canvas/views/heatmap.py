from graphlab.canvas.views.bivariate import BivariateView as _bivariate

import graphlab as gl

class HeatmapView(_bivariate):

    #maximum # of rows in individually processed SFrame chunks
    MAX_CHUNK_SIZE = 50000;
    #sample size from which to draw initial extrema
    INITIAL_EXTREMA_SAMPLE_SIZE = 1000;
    #(sample size to probe to find initial extrema upper bound)
    PROBING_SAMPLE_SIZE = 1000000;

    def __init__(self, obj, params):
        # obj is an SFrame
        # col1 is the name of the column in the X axis
        # col2 is the name of the column in the Y axis
        super(HeatmapView, self).__init__(obj, params)
        self.col1 = self.get_numeric_column('x')
        self.col2 = self.get_numeric_column('y', self.col1)
        self.__re_init()

    def __re_init(self):
        # TODO this is a hack -- we should better keep track of which columns
        # are currently being used.
        self.__heatmap_col1 = self.col1
        self.__heatmap_col2 = self.col2
        if self.col1 is None or \
           self.col2 is None:
            return
        self.__heatmap = gl.extensions._canvas.streaming.heatmap()
        self.__heatmap.init(self.obj[self.col1], self.obj[self.col2])

    def get_staticdata(self):
        # same as _get_bins but synchronous, for IPython/Jupyter Notebook output
        return self.__get_bins(True)

    #called to return all the values. When dealing with large datasets, called periodically while binning occurs
    def _get_bins(self, url, handler):
        handler.write(self.__get_bins(False))

    def __get_bins(self, materialize):
        if self.col1 is None or \
           self.col2 is None:
            handler.write({})
            return

        if self.col1 != self.__heatmap_col1 or \
           self.col2 != self.__heatmap_col2:
            self.__re_init()

        val = self.__heatmap.get()
        if materialize:
            while not(self.__heatmap.eof()):
                val = self.__heatmap.get()

        bins = list(map(list, val.bins)) # TODO don't JSON serialize lots of numbers
        BINS_ON_SIDE = float(val.num_bins)
        return {
            'bins': bins,
            'binWidth': (val.bin_extrema.x.max - val.bin_extrema.x.min) / BINS_ON_SIDE,
            'binHeight': (val.bin_extrema.y.max - val.bin_extrema.y.min) / BINS_ON_SIDE,
            'extrema': {
                'maxX': val.bin_extrema.x.max,
                'maxY': val.bin_extrema.y.max,
                'minX': val.bin_extrema.x.min,
                'minY': val.bin_extrema.y.min
            },
            'domainToShow': {
                'maxX': val.extrema.x.max,
                'maxY': val.extrema.y.max,
                'minX': val.extrema.x.min,
                'minY': val.extrema.y.min
            },
            'pointsLoaded': self.__heatmap.rows_processed,
            'progress': float(self.__heatmap.rows_processed) / len(self.obj),
            'complete': self.__heatmap.eof()
        }
