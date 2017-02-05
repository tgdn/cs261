import graphlab
import graphlab.canvas
import graphlab.canvas.views.bar
import graphlab.canvas.views.base
import graphlab.canvas.views.boxwhisker
import graphlab.canvas.views.heatmap
import graphlab.canvas.views.sarray
import graphlab.canvas.views.scatter
import graphlab.connect as _mt

import json
import re
import sys

class SFrameView(graphlab.canvas.views.base.BaseView):

    def __init__(self, obj, params=None):
        super(SFrameView, self).__init__(obj, params)

        # metrics
        _mt._get_metric_tracker().track('canvas.sframe.row.size', value=len(obj))
        _mt._get_metric_tracker().track('canvas.sframe.column.size', value=len(obj.column_names()))

        # initialize members
        self.__child_views = {}
        self.__child_identifiers = {}

        # initialize plot views
        self.scatter = graphlab.canvas.views.scatter.ScatterPlot(self.obj, self.params)
        self.heatmap = graphlab.canvas.views.heatmap.HeatmapView(self.obj, self.params)
        self.bar = graphlab.canvas.views.bar.BarGraphView(self.obj, self.params)
        self.boxwhisker = graphlab.canvas.views.boxwhisker.BoxWhiskerView(self.obj, self.params)

        # register SFrame-wide (Table, Summary) handlers
        self.register_handler('post', 'plot_params', self.__post_plot_params)
        self.register_handler('post', 'columns', self.__post_columns)
        self.register_handler('get', 'rows', self.__get_rows)
        self.register_handler('get', 'sketch/.*', self.__get_sketch)
        self.register_handler('get', 'cancel_sketch/.*', self.__cancel_sketch)
        self.register_handler('get', 'cached_sketches', self.__get_cached_sketches)

        # handlers for Scatter view
        self.register_handler('get', 'scatter_values', self.scatter._get_values)

        # handlers for Heat Map
        self.register_handler('get', 'bins', self.heatmap._get_bins)

        # handlers for BarGraph view
        self.register_handler('get', 'bar_values', self.bar._get_values)

        # handlers for BoxWhisker view
        self.register_handler('get', 'boxwhisker_values', self.boxwhisker._get_values)

        # if plot picked columns but were not specified in params, put into params
        if 'view' in self.params and self.params['view'] is not None:
            view = self.__get_view_instance()
            if view is not None:
                if not('x' in self.params) or self.params['x'] is None:
                    self.params['x'] = view.col1
                if not('y' in self.params) or self.params['y'] is None:
                    self.params['y'] = view.col2

        # materialize this sframe so __get_content_identifier__ returns valid results
        self.obj.__materialize__()

    def __check_identifier(self, col):
        # need to check identifier against real column (not cached)
        # in case the column itself has changed
        sa = self.obj[col]

        # do not check identifiers on special SGraph columns
        # they are different each time.
        import graphlab.data_structures.gframe # import here instead of at file-level to avoid circular dependency
        if isinstance(sa, graphlab.data_structures.gframe.GFrame):
            if not(col in self.__child_identifiers):
                self.__child_identifiers[col] = 0
                self.__child_views[col] = graphlab.canvas.views.sarray.SArrayView(sa)
            return

        ci = sa.__get_content_identifier__()
        if not(col in self.__child_identifiers):
            self.__child_identifiers[col] = ci
            self.__child_views[col] = graphlab.canvas.views.sarray.SArrayView(sa)
        elif self.__child_identifiers[col] != ci:
            # mismatch, delete cached objects
            self.__child_identifiers[col] = ci
            self.__child_views[col] = graphlab.canvas.views.sarray.SArrayView(sa)

    def __post_columns(self, url, handler):
        columns = handler.get_argument('columns')
        columns = list(map(str, json.loads(columns)))

    def __get_view_instance(self):
        views = {
            'Scatter Plot': self.scatter,
            'Heat Map': self.heatmap,
            'Bar Chart': self.bar,
            'BoxWhisker Plot': self.boxwhisker,
            'Line Chart': self.scatter
        }
        if self.params['view'] in views:
            return views[self.params['view']]
        return None

    def __post_plot_params(self, url, handler):
        view = handler.get_argument('view')
        col1 = handler.get_argument('col1').encode('utf-8')
        col2 = handler.get_argument('col2').encode('utf-8')
        if sys.version_info.major > 2:
            col1 = col1.decode()
            col2 = col2.decode()

        self.params['view'] = view
        plot = self.__get_view_instance()
        plot.col1 = self.params['x'] = col1
        plot.col2 = self.params['y'] = col2

    def __expand_columns(self):
        # given an SFrame, expands columns into a structure like:
        # {'str': ['col1', 'col3'], 'int': ['col2']}
        columns = []
        for (name,dtype) in zip(self.obj.column_names(), self.obj.column_types()):
            columns.append({
                'name': name,
                'dtype': dtype.__name__
            })
        return columns

    def __get_row_values(self, start, end, columns):
        # build a table (2d-array) of values,
        # like orient='values' in Pandas to_json
        # except we will substitute placeholder string values for
        # vector types (array, list, dict)
        sf = self.obj[columns][start:end]
        return [[graphlab.canvas.views.base._encode_value(row[col]) \
                for col in columns] \
                for row in sf]

    def __get_rows(self, url, handler):
        m = re.match('rows/(.*)/(.*)', url)
        columns = self.obj.column_names()
        start = int(m.group(1))
        end = int(m.group(2))
        handler.write({
            'values': self.__get_row_values(start, end, columns)
        })

    def __get_sketch(self, url, handler):
        m = re.match('sketch/(.*)', url)
        col = m.group(1).encode('utf-8')
        childView = self.child_views()[col.decode()]
        handler.write(childView.get_sketch())

    def __cancel_sketch(self, url, handler):
        m = re.match('cancel_sketch/(.*)', url)
        col = m.group(1).encode('utf-8')
        childView = self.child_views()[col]
        childView.cancel_sketch()
        handler.write({})

    def __get_cached_sketches(self, url, handler):
        child_views = self.child_views()
        handler.write({col: child_views[col].get_sketch(cached_only=True) for col in self.obj.column_names()})

    def __update_child_views(self):
        column_names = set(self.obj.column_names())
        # check existing or generate new cached identifiers and child views
        for c in column_names:
            self.__check_identifier(c)

    def child_views(self):
        # lazily initialize SArrayView objects for children and store them here
        # so that we can share cached sketches with SArrayView
        self.__update_child_views()
        return self.__child_views

    def get_metadata(self):
        self.__update_child_views()
        self.params['columns'] = self.obj.column_names()
        return {
            'descriptives': {
                'rows': len(self.obj),
                'columns': len(self.obj.column_names())
            },
            'columns': self.__expand_columns(),
            'column_identifiers': list(self.__child_identifiers),
            'view_params': self.params
        }

    def get_staticdata(self):
        self.__update_child_views()
        columns = self.obj.column_names()
        data = {
            'columns': self.__expand_columns()
        }
        if self.get_js_component() == 'Summary':
            data.update({
                'sketch': dict(map(lambda x: (x, self.__child_views[x].get_sketch(background=False)), columns))
            })
        elif self.get_js_component() == 'Plots':
            if self.params['view'] == 'Heat Map':
                data.update(self.heatmap.get_staticdata())
            elif self.params['view'] == 'Scatter Plot' or \
                 self.params['view'] == 'Line Chart':
                data.update(self.scatter.get_staticdata())
            elif self.params['view'] == 'Bar Chart':
                data.update(self.bar.get_staticdata())
            elif self.params['view'] == 'BoxWhisker Plot':
                data.update(self.boxwhisker.get_staticdata())
        return data

    def get_js_file(self):
        return 'sframe'

    def get_js_component(self):
        if 'view' in self.params:
            import logging
            name = self.params['view']
            if name is None: #show summary view by Default
                return 'Summary'
            if name == 'Summary':
                return name
            if name == 'Table':
                if self._in_ipython():
                    logging.warning('%s view of SFrame is not supported in ipynb envionment. Fall back to Summary View. ' % name)
                    return 'Summary'
                else :
                    return 'Table'
            if name in self.get_js_components():
                return 'Plots'
            # in case of other random strings
            logging.warning('View param %s is not recognized. You may want to use "Summary", "Plots", "Table" or a specific plot type such as "Scatter Plot". I am showing you the Summary view for now.' % name)
        return 'Summary'

    def get_js_components(self):
        return [
            'Summary',
            'Table',
            'Bar Chart',
            'BoxWhisker Plot',
            'Line Chart',
            'Scatter Plot',
            'Heat Map',
            'Plots'
        ]

    def validate_js_component_name(self, name):
        super(SFrameView, self).validate_js_component_name(name)
        if self._in_ipython():
            if name == 'Table':
                raise ValueError('%s view of SFrame is not supported in ipynb target.' % name)
