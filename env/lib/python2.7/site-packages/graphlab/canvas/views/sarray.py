import graphlab.canvas.views.base
import graphlab.connect as _mt

import array
import datetime
import math
import operator
import re
import threading
import six
import sys

def _frange(start, stop, step):
    """ Helper: Range function that can take float start/stop/step """
    while start < stop:
        yield start
        start += step

def _compose(f, g):
    # returns f of g as a function
    def f_of_g(arg):
        return f(g(arg))
    return f_of_g

def _sanitize_number(n):
    # get rid of non-JSON-serializable numeric values
    if n == float('inf') or \
       n == float('-inf') or \
       math.isnan(n):
      return None
    return n

class SArrayView(graphlab.canvas.views.base.BaseView):

    def __init__(self, obj, params=None):
        super(SArrayView, self).__init__(obj, params)
        _mt._get_metric_tracker().track('canvas.sarray.row.size', value=len(obj))
        self.register_handler('get', 'rows', self.__get_rows)
        self.register_handler('get', 'sketch', self.__get_sketch)
        self.register_handler('get', 'cancel_sketch', self.__cancel_sketch)
        self.register_handler('get', 'subsketch', self.__get_subsketch)
        self.register_handler('get', 'histogram', self.__get_histogram)
        self.__sketch_lock = threading.RLock()
        self.__cached_sketch = None
        self.__cached_sketch_json = None
        self.__cached_sketch_with_subsketches = None
        self.__cached_histogram = None

    def get_metadata(self):
        ret = {
            'descriptives': {
                'rows': len(self.obj)
            },
            'dtype': self.obj.dtype().__name__
        }
        if issubclass(self.obj.dtype(), array.array):
            ret.update({
                'max_subcolumn_length': self.__max_element_length()
            })
        return ret

    def get_staticdata(self):
        from graphlab.data_structures.image import Image
        if issubclass(self.obj.dtype(), Image):
            return {
                'rows': self.__get_row_values(0, 20)
            }
        return {
            'sketch': self.get_sketch(background=False),
            'histogram': self.__get_histogram_static()
        }

    def __get_row_values(self, start, end):
        # build a table (2d-array) of values,
        # like orient='values' in Pandas to_json
        # except we will substitute placeholder string values for
        # vector types (array, list, dict)
        return [graphlab.canvas.views.base._encode_value(row) \
                for row in self.obj[start:end]]

    def __get_rows(self, url, handler):
        m = re.match('rows/(.*)/(.*)', url)
        start = int(m.group(1))
        end = int(m.group(2))
        handler.write({
            'values': self.__get_row_values(start, end)
        })

    def __get_sketch(self, url, handler):
        """
        Gives a sketch summary for this column.
        """
        handler.write(self.get_sketch())

    def __cancel_sketch(self, url, handler):
        """
        Cancels the sketch summary if it is running in the background.
        """
        self.cancel_sketch()
        handler.write({})

    def __max_element_length(self):
        """
        If this view wraps an SArray with dtype array.array,
        this will return the maximum length
        """
        with self.__sketch_lock:
            if self.__cached_sketch is not None:
                return int(self.__cached_sketch.element_length_summary().max())
            return 0

    def __get_subsketch(self, url, handler):
        """
        Gives a sketch summary for the sub-column (if this column is a dict type)
        """
        subcol = url.split('/')[1]
        with self.__sketch_lock:
          sk = self.__cached_sketch
          dtype = self.obj.dtype()
          if dtype == array.array:
              subcol = int(subcol)
          if not(sk.sketch_ready()):
              return {
                  'progress': SArrayView.__sketch_progress(sk),
                  'complete': False
              }
          if self.__cached_sketch_with_subsketches is None:
              # figure out which subsketch keys to use
              sub_sketch_keys = None
              if dtype == dict:
                  sorted_frequent_items = sorted(six.iteritems(sk.dict_key_summary().frequent_items()), key=operator.itemgetter(1), reverse=True)
                  sorted_frequent_keys = list(map(operator.itemgetter(0), sorted_frequent_items))
                  sub_sketch_keys = sorted_frequent_keys[:50] # keep this limit in sync with maxValues in DictView
              elif dtype == array.array:
                  sub_sketch_keys = range(self.__max_element_length())

              # create the sketch with subsketches
              self.__cached_sketch_with_subsketches = self.obj.sketch_summary(
                  background=True,
                  sub_sketch_keys=sub_sketch_keys
                  )

          # get element sub sketch and write response
          subsketch = self.__cached_sketch_with_subsketches.element_sub_sketch(subcol)
          ret = SArrayView.__format_sketch(subsketch)
          handler.write(ret)

    def __get_histogram(self, url, handler):
        """
        Gives a streaming histogram for the column.
        """
        if self.__cached_histogram is None:
            self.__cached_histogram = graphlab.extensions._canvas.streaming.histogram.continuous()
            self.__cached_histogram.init(self.obj)
        result = self.__cached_histogram.get()
        rows_processed = self.__cached_histogram.rows_processed
        histogram = result.get_bins(12)
        handler.write({
            'complete': self.__cached_histogram.eof(),
            'progress': float(rows_processed) / float(len(self.obj)),
            'histogram': {
                'bins': histogram.bins,
                'min': histogram.min,
                'max': histogram.max
            },
            'min': result.min,
            'max': result.max
        })

    def __get_histogram_static(self):
        """
        Computes the whole histogram ahead of time.
        """
        if self.obj.dtype() not in (int, float):
            return None

        if self.__cached_histogram is None:
            self.__cached_histogram = graphlab.extensions._canvas.streaming.histogram.continuous()
            self.__cached_histogram.init(self.obj)
        result = self.__cached_histogram.get()
        while not(self.__cached_histogram.eof()):
            result = self.__cached_histogram.get()
        rows_processed = self.__cached_histogram.rows_processed
        histogram = result.get_bins(12)
        return {
            'complete': self.__cached_histogram.eof(),
            'progress': float(rows_processed) / float(len(self.obj)),
            'histogram': {
                'bins': histogram.bins,
                'min': histogram.min,
                'max': histogram.max
            },
            'min': result.min,
            'max': result.max
        }

    @staticmethod
    def __sketch_progress(sk):
        if sk is None:
            return 0.
        return float(sk.num_elements_processed()) / float(sk.size()) if sk.size() > 0 else 1.

    @staticmethod
    def __format_frequent_items(items):
        # format values using _encode_value, do not truncate key
        ret = {}
        from graphlab.data_structures.image import Image
        for k,v in six.iteritems(items):
            key = k
            if isinstance(key, (datetime.datetime, Image)):
                key = str(id(key)) # convert datetime, image to str first (keys must be string)
            if type(key) == str and sys.version_info.major == 2:
                key = unicode(key, errors='replace')
            ret[key] = {
                'frequency': graphlab.canvas.views.base._encode_value(v),
                'value': graphlab.canvas.views.base._encode_value(k)
            }
        return ret

    @staticmethod
    def __format_sketch(sk):
        # if the sketch is not ready, just wait, don't format yet
        # the client will poll again
        if not(sk.sketch_ready()):
            return {
                'progress': SArrayView.__sketch_progress(sk),
                'complete': False
            }

        numeric = False
        try:
            sk.min()
            numeric = True
        except:
            # TODO is there a better way to detect whether the sketch is over
            # numeric data than to use exceptions for control flow?
            pass
        data = {
            'numeric': numeric,
            'size': sk.size(),
            'num_undefined': sk.num_undefined(),
            'num_unique': sk.num_unique(),
            'frequent_items': SArrayView.__format_frequent_items(sk.frequent_items()),
            'progress': SArrayView.__sketch_progress(sk),
            'complete': sk.sketch_ready()
        }
        if numeric:
            try:
              data.update({
                'min': _sanitize_number(sk.min()),
                'max': _sanitize_number(sk.max()),
                'mean': _sanitize_number(sk.mean()),
                'median': _sanitize_number(sk.quantile(0.5)),
                'var': _sanitize_number(sk.var()),
                'std': _sanitize_number(sk.std()),
                'quantile': list(map(_compose(_sanitize_number, sk.quantile), _frange(0, 1.01, 0.01))) if sk.size() > 0 else []
              })
            except:
                # empty dictionary columns will throw an exception for sk.quantile(0.5) call
                pass
        return data

    def get_sketch(self, background=True, cached_only=False):
        """
        Returns a dictionary representation of a sketch summary. For vector
        types, this will collapse the summary down to an element-wise or
        value-wise (aggregate) summary.

        Parameters
        ----------
        background : bool
            Run the sketch on a background thread (if not already started) and
            return results immediately.

        cached_only : bool
            If there has not yet been a sketch initiated, do not initiate one
            (and return None).
        """
        with self.__sketch_lock:
            sa = self.obj

            # bypass sketch_summary calls if the sketch is complete
            if self.__cached_sketch_json is not None:
                return self.__cached_sketch_json
            elif cached_only:
                return None

            ret = None
            from graphlab.data_structures.image import Image
            if issubclass(self.obj.dtype(), Image):
                ret = {
                    'progress': 1.,
                    'complete': True,
                    'num_undefined': self.obj.num_missing(),
                    'samples': self.__get_row_values(0,4)
                }
            else:
                if self.__cached_sketch is None:
                    self.__cached_sketch = sa.sketch_summary(background=background)
                sk = self.__cached_sketch
                if issubclass(sa.dtype(), dict):
                    key_sketch = SArrayView.__format_sketch(sk.dict_key_summary())
                    value_sketch = SArrayView.__format_sketch(sk.dict_value_summary())
                    progress = (
                        key_sketch['progress'],
                        value_sketch['progress'],
                        SArrayView.__sketch_progress(sk)
                    )
                    complete = all((
                        key_sketch['complete'],
                        value_sketch['complete'],
                        sk.sketch_ready()
                    ))
                    ret = {
                        'keys': key_sketch,
                        'values': value_sketch,
                        'progress': sum(progress)/len(progress),
                        'complete': complete,
                        'num_undefined': sk.num_undefined()
                    }
                elif issubclass(sa.dtype(), (list, array.array)):
                    ret = SArrayView.__format_sketch(sk.element_summary())
                    progress = (ret['progress'], SArrayView.__sketch_progress(sk))
                    ret['progress'] = sum(progress)/len(progress)
                    ret['complete'] = all((ret['complete'], sk.sketch_ready()))
                else:
                    ret = SArrayView.__format_sketch(sk)

            # if we are done with the sketch, save the dict representation
            # and remove the Sketch object
            if ret['complete']:
                self.__cached_sketch_json = ret
                if not(issubclass(sa.dtype(), (array.array, dict))):
                    # clear the cached sketch, unless this is a dict
                    # or array column, in which case we need it later
                    # on for computing subsketches w/ keys
                    self.__cached_sketch = None
            return ret

    def cancel_sketch(self):
        """
        Cancels an in-progress sketch_summary.
        """
        with self.__sketch_lock:
            if self.__cached_sketch is not None:
                self.__cached_sketch.cancel()
                self.__cached_sketch = None
                self.__cached_sketch_json = None
                self.__cached_sketch_with_subsketches = None

    def get_js_file(self):
        return 'sarray'

    def get_js_component(self):
        if 'view' in self.params and \
            self.params['view'] is not None:
            name = self.params['view']
            self.validate_js_component_name(name)
            return name
        return self.get_js_components()[0] # default to 1st available tab

    def get_js_components(self):
        from graphlab.data_structures.image import Image
        return {
            str: ['Categorical'],
            int: ['Numeric', 'Categorical'],
            float: ['Numeric', 'Categorical'],
            dict: ['Dictionary'],
            array.array: ['Array'],
            list: ['List'],
            datetime.datetime: ['Categorical'],
            Image: ['Images']
        }[self.obj.dtype()]
