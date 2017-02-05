'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''
from .. import connect as _mt
import graphlab as _graphlab
import datetime as _datetime
import logging as _logging

from graphlab import glconnect as _glconnect
from graphlab.util import _is_non_string_iterable, _make_internal_url

from . import interpolation as _interpolation
from ._grouped_timeseries import GroupedTimeSeries
from collections import namedtuple as _namedtuple
from graphlab.util import _raise_error_if_not_of_type
from ..deps import pandas, HAS_PANDAS, numpy, HAS_NUMPY

##############################################################################
#
#                     TimeSeries helper utilities
#
##############################################################################

def _parse_interpolator_inputs(upsample_method):
    up_op = None
    import inspect
    if inspect.isfunction(upsample_method):
        try:
            up_op, col = upsample_method()
        except:
            raise TypeError(
                    "Unexpected type in the upsample_method definition.")
    else:
      if type(upsample_method) == str:
          if upsample_method not in TimeSeries._TIMESERIES_INTERPOLS:
            raise ValueError(
                  "upsample_method must one of the following: %s" \
                     % ','.join(TimeSeries._TIMESERIES_INTERPOLS.keys()))

          up_op, col = TimeSeries._TIMESERIES_INTERPOLS[upsample_method]()
      elif type(upsample_method) == tuple:
          up_op, col = upsample_method
      elif upsample_method == None:
          up_op, col = _interpolation.NONE()
      else:
        raise TypeError(
           "Unexpected type in the downsample_method definition.")
    return up_op

def _parse_downsample_inputs(downsample_method, value_col_names,
        index_col_name):
    agg_operators = None
    if type(downsample_method) == str:
        if downsample_method not in TimeSeries._TIMESERIES_AGGS:
          raise ValueError(
                "downsample_method must one of the following: %s" \
                    % ','.join(TimeSeries._TIMESERIES_AGGS.keys()))

        agg_operators = {}
        for col in value_col_names:
          if downsample_method in ['argmax', 'argmin']:
              agg_operators[col] =\
                TimeSeries._TIMESERIES_AGGS[downsample_method](col,
                        index_col_name)
          else:
              agg_operators[col] =\
                TimeSeries._TIMESERIES_AGGS[downsample_method](col)
        agg_operators = [agg_operators]

    else:
        agg_operators = [downsample_method]
    return agg_operators

def _parse_aggregator_inputs(agg_operators, column_types):
    ds_output_columns = []
    ds_columns = []
    ds_ops = []
    for op_entry in agg_operators:
        operation = op_entry
        if not(isinstance(operation, list) or isinstance(operation, dict)):
          operation = [operation]
        if isinstance(operation, dict):
          for key in operation:
              val = operation[key]
              if type(val) is tuple:
                (op, column) = val
                if op in ['__builtin__argmax__', '__builtin__argmin__']:
                    if (type(column[0]) is tuple) != (type(key) is tuple):
                      raise TypeError(
                        "The output column(s) and aggregate column(s) " +
                        " parameters must be of same type (tuple or string).")

                if (op == '__builtin__argmax__' or op == '__builtin__argmin__') \
                                         and type(column[0]) is tuple:
                  for (col,output) in zip(column[0],key):
                    ds_columns += [[col, column[1]]]
                    ds_ops += [op]
                    ds_output_columns += [key]
                else:
                  ds_columns += [column]
                  ds_ops += [op]
                  ds_output_columns += [key]

                if op == '__builtin__concat__dict__':
                    key_column = column[0]
                    key_column_type = column_types[key_column]
                    if not key_column_type in (int, float, str):
                        raise TypeError(
                    'CONCAT key column must be int, float or str type')

              elif val == aggregate.COUNT:
                (op, column) = aggregate.COUNT()
                ds_output_columns += [key]
                ds_columns = ds_columns + [column]
                ds_ops = ds_ops + [op]
              else:
                raise TypeError("Unexpected type in the definition of "
                                  " the output column: " + key)

        elif isinstance(operation, list):
          for val in operation:
              if type(val) is tuple:
                (op, column) = val
                if op in ['__builtin__argmax__', '__builtin__argmin__'] and\
                                 type(column) is tuple:
                  for col in column[0]:
                    ds_columns += [[col, column[1]]]
                    ds_ops += [op]
                    ds_output_columns += [""]
                else:
                  ds_columns += [column]
                  ds_ops += [op]
                  ds_output_columns += [""]

                if op == '__builtin__concat__dict__':
                    key_column = column[0]
                    key_column_type = column_types[key_column]
                    if not key_column_type in (int, float, str):
                        raise TypeError(
                        'CONCAT key column must be int, float or str type')

              elif val == _graphlab.aggregate.COUNT:
                val = _graphlab.aggregate.COUNT()
                (op, column) = val
                ds_output_columns += [""]
                ds_columns += [column]
                ds_ops += [op]
              else:
                raise TypeError(
                        "Unexpected type in the aggregator definition.")
        else:
          raise TypeError(str(operation) + " is of unexpected type.")

    for (cols, op) in zip(ds_columns, ds_ops):
        for col in cols:
            if not isinstance(col, str):
                raise TypeError("Unexpected type for parameter " +
                         col + " (expecting string)")
        if not isinstance(op, str):
                raise TypeError("Unexpected type for parameter " + op)

    return ds_output_columns, ds_columns, ds_ops

##############################################################################
#
#                     TimeSeries definition & methods.
#
##############################################################################
_MAX_ROWS_TO_DISPLAY = 10
_FOOTER_STRS = ['Note: Only the head of the TimeSeries is printed.',
               'You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.']

__LOGGER__ = _logging.getLogger(__name__)

def load_timeseries(location):
    '''
    Load a saved TimeSeries object from a given location.

    Parameters
    ----------
    location : str
        The path to load the TimeSeries object. The path can be a location in
        the file system, an s3 path, or a HDFS path.

    See Also
    --------
    TimeSeries.save

    Examples
    --------
    .. sourcecode:: python

      >>> import graphlab as gl
      >>> import datetime as dt

      >>> index = gl.SArray.date_range(start_time = dt.datetime(2015, 1, 1),
      ...                              end_time = dt.datetime(2016, 1, 1),
      ...                              freq = dt.timedelta(days = 1))
      >>> value = gl.SArray(range(len(index)))
      >>> data = gl.SFrame({'index': index, 'value': value})

      # Create a time-series.
      >>> ts = gl.TimeSeries(data, index='index')

      # Save the time-series to disk (a directory with the name is created)
      >>> ts.save("./my_series.ts")

      # Load the time-series.
      >>> loaded_ts = gl.load_timeseries("./my_series")
      >>> print loaded_ts
      +---------------------+-------+
      |        index        | value |
      +---------------------+-------+
      | 2015-01-01 00:00:00 |   0   |
      | 2015-01-02 00:00:00 |   1   |
      | 2015-01-03 00:00:00 |   2   |
      | 2015-01-04 00:00:00 |   3   |
      | 2015-01-05 00:00:00 |   4   |
      | 2015-01-06 00:00:00 |   5   |
      | 2015-01-07 00:00:00 |   6   |
      | 2015-01-08 00:00:00 |   7   |
      | 2015-01-09 00:00:00 |   8   |
      | 2015-01-10 00:00:00 |   9   |
      +---------------------+-------+
      [366 rows x 2 columns]
    '''
    _mt._get_metric_tracker().track('graphlab.load_timeseries')
    if not isinstance(location,str):
        raise TypeError("The `location` argument must be from type str.")
    _gl_timeseries = _graphlab.load_model(location)
    return TimeSeries(_gl_timeseries.sframe, _gl_timeseries.index_col_name,
                      gl_timeseries=_gl_timeseries)


class TimeSeries(object):
    '''
    The TimeSeries object is the fundamental data structure for multivariate
    time series data. TimeSeries objects are backed by a single `SFrame`, but
    include extra metadata.

    The TimeSeries data is stored like the following:

    ====== ====== ====== ===== ======
       T     V_0    V_1   ...    V_n
    ====== ====== ====== ===== ======
      t_0   v_00   v_10   ...   v_n0
      t_1   v_01   v_11   ...   v_n1
      t_2   v_02   v_12   ...   v_n2
      ...   ...    ...    ...   ...
      t_k   v_0k   v_1k   ...   v_nk
    ====== ====== ====== ===== ======

    Each column in the table is a univariate time series, and the index is
    shared across all of the series.

    Parameters
    ----------
    data : SFrame | str
        ``data`` is either the SFrame that holds the content of the
        TimeSeries object or is a string. If it is a string, it is
        interpreted as a filename. Files can be read from local file system
        or a URL (local://, hdfs://, s3://, http://).

    index : str
        The name of the column containing the index of the time series in
        the SFrame referred to by ``data``.  The column must be of type
        datetime.datetime.  ``data`` will be sorted by the ``index`` column
        if it is not already sorted by this column.  If ``data`` is a
        filename, this parameter is optional and ignored; otherwise it is
        required.

    **kwargs : optional
        Keyword parameters passed to the TimeSeries constructor.

        - *is_sorted* : bool, optional

    Examples
    --------

    **Construction**

    .. sourcecode:: python

        >>> import graphlab as gl
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                 'b': [t0, t0 + dt.timedelta(days=5),
        ...                       t0 + dt.timedelta(days=10)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> print ts
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

    **Save and Load**

    .. sourcecode:: python

        >>> ts.save("my_series")
        >>> ts_new = gl.TimeSeries("my_series")
        >>> print ts_new
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

    **Element Accessing in TimeSeries**

    .. sourcecode:: python

        >>>  ts.index_col_name
        'b'
        >>> ts.value_col_names
        ['a']
        >>> ts['a']
        dtype: float
        Rows: 3
        [1.1, 2.1, 3.1]
        >>> ts[0]
        {'a': 1.1, 'b': datetime.datetime(2013, 5, 7, 10, 4, 10)}

    **Resampling TimeSeries**

    .. sourcecode:: python

        >>> t_resample = ts.resample(dt.timedelta(days=1),
        ...                          downsample_method='sum',
        ...                          upsample_method='nearest')
        >>> print t_resample
        e
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 00:00:00 | 1.1 |
        | 2013-05-08 00:00:00 | 1.1 |
        | 2013-05-09 00:00:00 | 1.1 |
        | 2013-05-10 00:00:00 | 2.1 |
        | 2013-05-11 00:00:00 | 2.1 |
        | 2013-05-12 00:00:00 | 2.1 |
        | 2013-05-13 00:00:00 | 2.1 |
        | 2013-05-14 00:00:00 | 2.1 |
        | 2013-05-15 00:00:00 | 3.1 |
        | 2013-05-16 00:00:00 | 3.1 |
        +---------------------+-----+
        [11 rows x 2 columns]

    **Shifting Index Column**

    .. sourcecode:: python

        >>> interval  = dt.timedelta(days=5)
        >>> ts_tshifted = ts.tshift(steps=interval)
        >>> print ts_tshifted
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-12 10:04:10 | 1.1 |
        | 2013-05-17 10:04:10 | 2.1 |
        | 2013-05-22 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

    **Shifting Value Columns**

    .. sourcecode:: python

        >>> ts_shifted = ts.shift(steps=2)
        >>> print ts_shifted
        +---------------------+------+
        |          b          |  a   |
        +---------------------+------+
        | 2013-05-07 10:04:10 | None |
        | 2013-05-12 10:04:10 | None |
        | 2013-05-17 10:04:10 | 1.1  |
        +---------------------+------+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

        >>> ts_shifted = ts.shift(steps=-1)
        >>> print ts_shifted
        +---------------------+------+
        |          b          |  a   |
        +---------------------+------+
        | 2013-05-07 10:04:10 | 2.1  |
        | 2013-05-12 10:04:10 | 3.1  |
        | 2013-05-17 10:04:10 | None |
        +---------------------+------+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

    **Join Two TimeSeries on Index Columns**

    .. sourcecode:: python

        >>> import graphlab as gl
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                 'b':[t0, t0 + dt.timedelta(days=1),
        ...                      t0 + dt.timedelta(days=2)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> print ts
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-08 10:04:10 | 2.1 |
        | 2013-05-09 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

        >>> sf2 = gl.SFrame({'a':[1.1, 2.1, 3.1],
        ...                  'b':[t0 + dt.timedelta(days=1),
        ...                       t0 + dt.timedelta(days=2),
        ...                       t0 + dt.timedelta(days=3)]})
        >>> ts2 = gl.TimeSeries(sf2, index='b')
        >>> print ts2
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-08 10:04:10 | 1.1 |
        | 2013-05-09 10:04:10 | 2.1 |
        | 2013-05-10 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

        >>> ts_join = ts.index_join(ts2, how='inner')
        >>> print ts_join
        +---------------------+-----+-----+
        |          b          |  a  | a.1 |
        +---------------------+-----+-----+
        | 2013-05-08 10:04:10 | 2.1 | 1.1 |
        | 2013-05-09 10:04:10 | 3.1 | 2.1 |
        +---------------------+-----+-----+
        [2 rows x 3 columns]
        The index column of the TimeSeries is: b

    **Slicing TimeSeries**

    .. sourcecode:: python

        >>> sliced_ts = ts.slice(t0, t0 + dt.timedelta(days=3),
        ...                               closed="left")
        >>> print sliced_ts
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        +---------------------+-----+
        [1 rows x 2 columns]
        The index column of the TimeSeries is: b

        >>> sliced_ts = ts[dt.date(2013, 5, 7)]
        >>> print sliced_ts
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        +---------------------+-----+
        [1 rows x 2 columns]
        The index column of the TimeSeries is: b

        >>> ts[dt.datetime(2013, 5, 7):dt.datetime(2013, 5, 13)]
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        +---------------------+-----+
        [2 rows x 2 columns]
        The index column of the TimeSeries is: b

    **Add/Remove TimeSeries Columns**

    .. sourcecode:: python

        >>> ts.add_column(gl.SArray([1, 2, 3]), "new_value")
        >>> print ts
        +---------------------+-----+-----------+
        |          b          |  a  | new_value |
        +---------------------+-----+-----------+
        | 2013-05-07 10:04:10 | 1.1 |     1     |
        | 2013-05-12 10:04:10 | 2.1 |     2     |
        | 2013-05-17 10:04:10 | 3.1 |     3     |
        +---------------------+-----+-----------+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: b

        >>> ts.remove_column("new_value")
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b
    '''

    _TIMESERIES_AGGS = {k.lower() : getattr(_graphlab.aggregate, k)
                for k in dir(_graphlab.aggregate) if not k.startswith('_')}
    _TIMESERIES_INTERPOLS = {k.lower() : getattr(_interpolation, k)
                for k in dir(_interpolation) if not k.startswith('_')}

    class date_part(object):
        """
        Values representing parts of a date.

        Used by ``group`` to specify parts of a date to group by. Each date
        part will be represented as a numeric value when grouping.

        Values supported:
            - YEAR: The year number
            - MONTH: A value between 1 and 12 where 1 is January.
            - DAY: Day of the months. Begins at 1.
            - HOUR: Hours since midnight.
            - MINUTE: Minutes after the hour.
            - SECOND: Seconds after the minute.
            - US: Microseconds after the second. Between 0 and 999,999.
            - WEEKDAY: A value between 0 and 6 where 0 is Monday.
            - ISOWEEKDAY: A value between 1 and 7 where 1 is Monday.
            - TMWEEKDAY: A value between 0 and 7 where 0 is Sunday
        """
        _year_class = _namedtuple("YEAR", ['name'])
        YEAR = _year_class(name="year")
        _month_class = _namedtuple("MONTH", ['name'])
        MONTH = _month_class(name="month")
        _day_class = _namedtuple("DAY", ['name'])
        DAY = _day_class(name="day")
        _hour_class = _namedtuple("HOUR", ['name'])
        HOUR = _hour_class(name="hour")
        _minute_class = _namedtuple("MINUTE", ['name'])
        MINUTE = _minute_class(name="minute")
        _second_class = _namedtuple("SECOND", ['name'])
        SECOND = _second_class(name="second")
        _weekday_class = _namedtuple("WEEKDAY", ['name'])
        WEEKDAY = _weekday_class(name="weekday")
        _isoweekday_class = _namedtuple("ISOWEEKDAY", ['name'])
        ISOWEEKDAY = _isoweekday_class(name="isoweekday")
        _tmweekday_class = _namedtuple("TMWEEKDAY", ['name'])
        TMWEEKDAY = _tmweekday_class(name="tmweekday")
        _us_class = _namedtuple("MICROSECOND", ['name'])
        MICROSECOND = _us_class(name="us")


    def __init__(self, data=None, index=None, **kwargs):
        '''
        Construct a new TimeSeries from a url or an SFrame.
        '''
        self.value_col_names = []
        self.index_col_name = None
        self._sframe = None
        self._is_sorted = False

        _mt._get_metric_tracker().track('timeseries.init')
        if (HAS_PANDAS and isinstance(data, pandas.DataFrame)) or\
             (HAS_PANDAS and isinstance(data, pandas.Series)):
            # This may be too restricting...not sure.
            # if not (HAS_NUMPY and data.index.dtype.type == numpy.datetime64):
            #    raise TypeError("Cannot convert to TimeSeries if index is not
            #    of type datetime64")
            data_index_added = data.reset_index()

            # Figure out which column was the index (it should just be called
            # 'index', but this seems more future proof, and can detect the
            # MultiIndex case)
            if index is None:
                if hasattr(data, 'columns'):
                    index_column = set(data_index_added.columns.values) \
                                                - set(data.columns.values)
                    if len(index_column) != 1:
                        raise TypeError(
                        "Cannot convert Pandas object with MultiIndex.")
                    index = index_column.pop()
                else:
                    # If there's no 'columns', then this is a Series, and
                    # there's no possibility of there being an existing column
                    # named "index"
                    index = 'index'

            # Convert Pandas dataframe to SFrame and let the logic below
            # convert to TimeSeries
            data = _graphlab.SFrame(data_index_added)

        if isinstance(data,str):
            self._timeseries = _graphlab.load_model(data)
            self.index_col_name = self._timeseries.index_col_name
            self.value_col_names = self._timeseries.value_col_names
        else:
            self._timeseries = kwargs.get('gl_timeseries')
            self._is_sorted = kwargs.get('is_sorted',False)

            if self._timeseries is not None:
                data = self._timeseries.sframe

            if not isinstance(data,_graphlab.SFrame):
                raise TypeError("The first argument must be an SFrame")
            if not isinstance(index,str):
                raise TypeError("Index argument must be str type.")

            column_names = data.column_names()
            if index is None:
                raise ValueError("Index column argument cannot be None")
            if index not in column_names:
                raise ValueError(
                    "Index column %s does not exist in the sframe." % index)
            self.index_col_name = index

            self.value_col_names = [val for val in column_names if val != index]

            if data[index].dtype() not in [_datetime.datetime] :
                raise TypeError(
                    "The index column '%s' must be of type datetime.datetime." \
                                     % self.index_col_name)

            if self._timeseries is None:
                self._timeseries = _graphlab.extensions._Timeseries()
                self._timeseries.init(data,index,self._is_sorted,[-1,-1])

        self._sframe = self._timeseries.sframe
        self.index = self._sframe[self.index_col_name]

    def __repr__(self):
        """
        Returns a string description of the TimeSeries.
        """
        ret = self.__get_column_description__()
        (is_empty, data_str) = self._sframe.__str_impl__()
        if is_empty:
            data_str = "\t[]"

        if self._sframe.__has_size__():
            ret = ret + "Rows: " + str(len(self)) + "\n\n"
        else:
            ret = ret + "Rows: Unknown" + "\n\n"

        ret = ret + "Data:\n"
        ret = ret + data_str
        return ret.replace('SFrame', 'TimeSeries')

    def __get_column_description__(self):
        colnames = self.column_names()
        coltypes = self.column_types()
        ret = "Columns:\n"
        if len(colnames) > 0:
            for i in range(len(colnames)):
                ret = ret + "\t" + colnames[i] + "\t" + coltypes[i].__name__
                if colnames[i] == self.index_col_name:
                    ret += " (index column)"
                ret += "\n"
            ret = ret + "\n"
        else:
            ret = ret + "\tNone\n\n"
        return ret

    def _repr_html_(self):
        return self._sframe._repr_html_().replace('SFrame', 'TimeSeries')

    def __str__(self, num_rows=10, footer=True):
        """
        Returns a string containing the first 10 elements of the TimeSeries.
        """
        sframe_str = self._sframe.__str__(num_rows, footer).replace(
                'SFrame', 'TimeSeries')
        ts_str =  "TimeSeries with '%s' as the index column." % self.index_col_name
        ts_str += '\n' + sframe_str
        return ts_str

    def __len__(self):
        '''
        Returns the number of rows of the TimeSeries.
        '''
        return self._sframe.__len__()

    def __copy__(self):
        """
        Returns a shallow copy of the TimeSeries.
        """
        return TimeSeries(self._sframe,index=self.index_col_name,is_sorted=True)


    def copy(self):
        """
        Returns a shallow copy of the TimeSeries.
        """
        return self.__copy__()

    def print_rows(self, num_rows=20, num_columns=40, max_column_width=30,
                   max_row_width=80):
        '''
        See :py:func:`~graphlab.SFrame.print_rows` for documentation.
        '''
        return self._sframe.print_rows(num_rows, num_columns, max_column_width,
                                       max_row_width)

    def column_names(self):
        '''
        See :py:func:`~graphlab.SFrame.column_names` for documentation.
        '''
        return self._sframe.column_names()

    def column_types(self):
        '''
        See :py:func:`~graphlab.SFrame.column_types` for documentation.
        '''
        return self._sframe.column_types()

    def argmax(self, agg_column):
        '''
        Return index of the row with the maximum value from agg_column.

        Parameters
        ----------
        agg_column : The name of the column to use as the value while finding
                     the argmax.

        Returns
        -------
        out : datetime.datetime

        Examples
        --------
        >>> print ts.argmax('stock_price')
        2013-05-17 10:04:10
        '''
        _mt._get_metric_tracker().track('timeseries.argmax')
        sf_out = self._sframe.groupby(key_columns=[],
           operations={'maximum_index':
                _graphlab.aggregate.ARGMAX(agg_column, self.index_col_name)})
        return sf_out['maximum_index'][0]

    def argmin(self, agg_column):
        '''
        Return index of the row with the minimum value from agg_column.

        Parameters
        ----------
        agg_column : The name of the column to use as the value while finding
                     the argmin.

        Returns
        -------
        out : datetime.datetime

        Examples
        --------
        >>> print ts.argmax('stock_price')
        2013-05-07 10:04:10
        '''
        _mt._get_metric_tracker().track('timeseries.argmin')
        sf_out = self._sframe.groupby(key_columns=[],
                operations={'minimum_index':_graphlab.aggregate.ARGMIN(agg_column,
                    self.index_col_name)})
        return sf_out['minimum_index'][0]

    def resample(self, period, downsample_method='avg', upsample_method=None,
                 label='left', closed='left'):
        '''
        Resample or bucketize the input TimeSeries based on the given period.

        This operator takes a potentially irregularly sampled TimeSeries as
        input and transforms it into to a TimeSeries with a regular frequency.

        There are three components in the resampling operator:

        * Mapping: Determining a time slice to which a particular timestamp
                   belongs to.
        * Upsampling/Interpolation: Determining what to do when no observations
                   map to a particular time slice.
        * Downsampling/Aggregation: Determining what to do when multiple
                   observations occur in the same time slice.


        Parameters
        ----------
        period : datetime.timedelta
            The frequency to which the output TimeSeries must be resampled.

        downsample_method : str | agg | dict[col:agg] | list[agg]
            Method for aggregating all values that occur within the time slice.

            The operators can be of type string (for shorthand),
            `~graphlab.aggregator`, list (of aggregators), or a dictionary
            (where the key is the output column name, and value is the
            aggregator).

            A string shorthand notation is available for applying the same
            aggregate on all columns (excluding for the index). For example,
            using downsample_method='avg' returns a timeseries with value
            columns containing the average of values occurring in the
            associated time bucket.

            Other operators include 'sum', 'max', 'min', 'count', 'avg', 'var',
            'stdv', 'concat', 'select_one', 'argmin', 'argmax', and 'quantile'.
            For more explicit use, aggregates can be invoked directly using the
            aggregates available in :mod:`~graphlab.aggregate` module. Note that
            for 'argmax' (and 'argmin'), the index value corresponding to the
            minimum (and maximum) value for each column is returned.


        upsample_method : inter | str
            Method for interpolation when a time slice does not contain any
            value.

            The operators can be of type string (for shorthand), or
            `~graphlab.timeseries.interpolate`. A string shorthand notation is
            available for interpolation operators.

            The following operators are currently availiable:

              - Forward fill (`~graphlab.interpolation.FFILL` or 'ffill')
              - Backward fill (`~graphlab.interpolation.BFILL` or 'bfill')
              - None (`~graphlab.interpolation.NONE` or 'none')
              - Nearest fill (`~graphlab.interpolation.NEAREST` or 'nearest')
              - Linear interpolation (`~graphlab.interpolation.linear` or 'linear')
              - Zero fill (`~graphlab.interpolation.ZERO` or 'zero')



        label : {'right', 'left'}, optional
            The timestamp recorded in the output TimeSeries to determine which
            end point (left or right) should be used to denote the time slice.
            For example, if the period is 1 hour, then `left` refers to each
            hour using its start time while `right` refers to each hour using
            its end time.

        closed : {'right', 'left'}, optional
            Determines which side of the interval in the time slice is closed.
            If `t` is the start time, and `t + T` is the end time of an
            interval (say 1 hour), then a `left` closed interval is [t, t + T)
            while a `right` closed interval is (t, t + T]

        Returns
        -------
        out : TimeSeries

        Examples
        --------

        .. sourcecode:: python

          >>> import datetime as dt
          >>> import graphlab as gl

          >>> household_data = gl.SFrame("https://static.turi.com/datasets/household_electric_sample.sf")
          >>> ts = gl.TimeSeries(household_data, index = 'DateTime')
          +---------------------+---------------------+-----------------------+---------+
          |       DateTime      | Global_active_power | Global_reactive_power | Voltage |
          +---------------------+---------------------+-----------------------+---------+
          | 2006-12-16 17:24:00 |        4.216        |         0.418         |  234.84 |
          | 2006-12-16 17:26:00 |        5.374        |         0.498         |  233.29 |
          | 2006-12-16 17:28:00 |        3.666        |         0.528         |  235.68 |
          | 2006-12-16 17:29:00 |         3.52        |         0.522         |  235.02 |
          | 2006-12-16 17:31:00 |         3.7         |          0.52         |  235.22 |
          | 2006-12-16 17:32:00 |        3.668        |          0.51         |  233.99 |
          | 2006-12-16 17:40:00 |         3.27        |         0.152         |  236.73 |
          | 2006-12-16 17:43:00 |        3.728        |          0.0          |  235.84 |
          | 2006-12-16 17:44:00 |        5.894        |          0.0          |  232.69 |
          | 2006-12-16 17:46:00 |        7.026        |          0.0          |  232.21 |
          +---------------------+---------------------+-----------------------+---------+
          [1025260 rows x 4 columns]

        The above event series has around 1M events spanning a 5 year period.
        The resample operator allows you to view the event series at various
        levels of granularity. For example, one can view the average daily
        values for each of the above columns as follows.

        .. sourcecode:: python

           >>> t_resample = ts.resample(dt.timedelta(days = 1),
           ...                          downsample_method='avg')
           +---------------------+---------------------+-----------------------+
           |       DateTime      | Global_active_power | Global_reactive_power |
           +---------------------+---------------------+-----------------------+
           | 2006-12-16 00:00:00 |    3.01369082126    |     0.086231884058    |
           | 2006-12-17 00:00:00 |    2.34746403385    |     0.159901269394    |
           | 2006-12-18 00:00:00 |    1.51999724138    |     0.109746206897    |
           | 2006-12-19 00:00:00 |    1.21253959732    |     0.108182550336    |
           | 2006-12-20 00:00:00 |     1.5366374829    |     0.108719562244    |
           | 2006-12-21 00:00:00 |    1.20097119342    |    0.0982825788752    |
           | 2006-12-22 00:00:00 |    1.59997759104    |     0.131694677871    |
           | 2006-12-23 00:00:00 |    3.33829255319    |     0.162789893617    |
           | 2006-12-24 00:00:00 |    1.77246153846    |     0.101983516484    |
           | 2006-12-25 00:00:00 |    1.92014035088    |     0.165074224022    |
           +---------------------+---------------------+-----------------------+
           +---------------+
           |    Voltage    |
           +---------------+
           | 236.289613527 |
           |  240.13448519 |
           | 241.308386207 |
           | 241.896845638 |
           | 242.404186047 |
           | 241.030781893 |
           | 241.331666667 |
           | 240.116954787 |
           | 241.776826923 |
           | 243.451336032 |
           +---------------+
           [1442 rows x 4 columns]

        Notice, that the output timeseries contains only a few thousand rows
        (one for each day in the 5 year period). If there are no values in the
        event series for a day, a `None` value is interpolated by default. This
        can be changed using the ``upsample_method`` argment. For example
        ``upsample_method=linear`` changes the interpolation scheme from
        filling in `None` values to a linear interpolation (of the previous and
        next days).

        .. sourcecode:: python

            >>> t_resample = ts.resample(dt.timedelta(days = 1),
            ...        downsample_method='avg', upsample_method = 'linear')
              +---------------------+---------------------+-----------------------+
              |       DateTime      | Global_active_power | Global_reactive_power |
              +---------------------+---------------------+-----------------------+
              | 2006-12-16 00:00:00 |    3.01369082126    |     0.086231884058    |
              | 2006-12-17 00:00:00 |    2.34746403385    |     0.159901269394    |
              | 2006-12-18 00:00:00 |    1.51999724138    |     0.109746206897    |
              | 2006-12-19 00:00:00 |    1.21253959732    |     0.108182550336    |
              | 2006-12-20 00:00:00 |     1.5366374829    |     0.108719562244    |
              | 2006-12-21 00:00:00 |    1.20097119342    |    0.0982825788752    |
              | 2006-12-22 00:00:00 |    1.59997759104    |     0.131694677871    |
              | 2006-12-23 00:00:00 |    3.33829255319    |     0.162789893617    |
              | 2006-12-24 00:00:00 |    1.77246153846    |     0.101983516484    |
              | 2006-12-25 00:00:00 |    1.92014035088    |     0.165074224022    |
              +---------------------+---------------------+-----------------------+
              +---------------+
              |    Voltage    |
              +---------------+
              | 236.289613527 |
              |  240.13448519 |
              | 241.308386207 |
              | 241.896845638 |
              | 242.404186047 |
              | 241.030781893 |
              | 241.331666667 |
              | 240.116954787 |
              | 241.776826923 |
              | 243.451336032 |
              +---------------+
              [1442 rows x 4 columns]


        By default, the output of a resample operator is a TimeSeries with a
        single time stamp that denotes the start of a time slice over which
        the aggregation/interpolation operations are performed. This labeling
        can be changed to the end using the `right` argument. This ensures
        that the buckets are labeled with the end of the time interval of
        the resample.

        .. sourcecode:: python

          >>> t_resample = ts.resample(dt.timedelta(days = 1),
          ...                          downsample_method='avg',
          ...                          upsample_method=None, label='right')

        We can also specify which side of the time slice is closed (default is
        'left').

        .. sourcecode:: python

          >>> t_resample = ts.resample(dt.timedelta(days = 1),
          ...                          downsample_method='avg',
          ...                          upsample_method=None,
          ...                          closed='right')


        ** Advanced Usage **

        We can also perform multiple operations on the same column. For
        example, in the following snippet, we calculate a daily average, and
        sum of the `Voltage` column side by side.

        .. sourcecode:: python

            >>> t_resample = ts.resample(dt.timedelta(days = 1),
            ...        downsample_method= [gl.aggregate.AVG('Voltage'),
            ...                            gl.aggregate.SUM('Voltage')])
            +---------------------+----------------+----------------+
            |       DateTime      | Avg of Voltage | Sum of Voltage |
            +---------------------+----------------+----------------+
            | 2006-12-16 00:00:00 | 236.289613527  |    48911.95    |
            | 2006-12-17 00:00:00 |  240.13448519  |   170255.35    |
            | 2006-12-18 00:00:00 | 241.308386207  |   174948.58    |
            | 2006-12-19 00:00:00 | 241.896845638  |   180213.15    |
            | 2006-12-20 00:00:00 | 242.404186047  |   177197.46    |
            | 2006-12-21 00:00:00 | 241.030781893  |   175711.44    |
            | 2006-12-22 00:00:00 | 241.331666667  |   172310.81    |
            | 2006-12-23 00:00:00 | 240.116954787  |   180567.95    |
            | 2006-12-24 00:00:00 | 241.776826923  |   176013.53    |
            | 2006-12-25 00:00:00 | 243.451336032  |   180397.44    |
            +---------------------+----------------+----------------+
            [1442 rows x 3 columns]

        Output column names can also be provided for the resample operation as
        follows. Here, we are returning 3 columns; `avg` denotes the average
        Voltage value for all events in the day, `min` and `max` respectivelt
        denote the minimum and maximum values for events within that day.

        .. sourcecode:: python

            >>> t_resample = ts.resample(dt.timedelta(days = 1),
            ...        downsample_method= {'avg': gl.aggregate.AVG('Voltage'),
            ...                            'min': gl.aggregate.MIN('Voltage'),
            ...                            'max': gl.aggregate.MAX('Voltage')})
            +---------------------+---------------+--------+--------+
            |       DateTime      |      avg      |  max   |  min   |
            +---------------------+---------------+--------+--------+
            | 2006-12-16 00:00:00 | 236.289613527 | 243.73 | 231.57 |
            | 2006-12-17 00:00:00 |  240.13448519 | 249.07 | 230.08 |
            | 2006-12-18 00:00:00 | 241.308386207 | 248.48 | 230.94 |
            | 2006-12-19 00:00:00 | 241.896845638 | 248.89 | 232.2  |
            | 2006-12-20 00:00:00 | 242.404186047 | 249.48 | 233.81 |
            | 2006-12-21 00:00:00 | 241.030781893 | 247.08 | 228.91 |
            | 2006-12-22 00:00:00 | 241.331666667 | 248.82 | 230.39 |
            | 2006-12-23 00:00:00 | 240.116954787 | 246.77 | 232.17 |
            | 2006-12-24 00:00:00 | 241.776826923 | 249.27 | 231.08 |
            | 2006-12-25 00:00:00 | 243.451336032 | 250.62 | 233.48 |
            +---------------------+---------------+--------+--------+
            [1442 rows x 4 columns]

        '''

        _raise_error_if_not_of_type(period, [_datetime.timedelta], 'period')
        _raise_error_if_not_of_type(label, [str], 'label')
        _raise_error_if_not_of_type(closed, [str], 'closed')
        _raise_error_if_not_of_type(upsample_method, [str, tuple, type(None)],
                'upsample_method')
        _raise_error_if_not_of_type(downsample_method, [str, list, dict, tuple],
                'downsample_method')
        if period == _datetime.timedelta():
            raise ValueError("Period must be non-zero")

        agg_operators = _parse_downsample_inputs(downsample_method,
                self._timeseries.value_col_names, self.index_col_name)
        ds_output_columns, ds_columns, ds_ops = _parse_aggregator_inputs(
                agg_operators, {k:v for k,v in \
                          zip(self.column_names(), self.column_types())})
        up_op = _parse_interpolator_inputs(upsample_method)

        # Log and call.
        _mt._get_metric_tracker().track('timeseries.resample',
                properties={'downsample_method': downsample_method,
                            'upsample_method': upsample_method,
                            'closed': closed,
                            'label': label})
        resampled_timeseries = self._timeseries.resample_wrapper(
                                      period.total_seconds(),
                                      [ds_columns, ds_output_columns, ds_ops],
                                      [up_op], label, closed)
        return TimeSeries(data=None,
                          index=self.index_col_name,
                          gl_timeseries = resampled_timeseries)

    def tshift(self, delta):
        '''
        Shift the index column of the TimeSeries object by 'delta' time.

        Parameters
        -----------
        delta : datetime.timedelta

        Returns
        --------
        out : TimeSeries

        Examples
        --------
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                 'b':[t0, t0 + dt.timedelta(days=5),
        ...                      t0 + dt.timedelta(days=10)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> print ts
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        >>> print ts.tshift(dt.timedelta(days=3))
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-10 10:04:10 | 1.1 |
        | 2013-05-15 10:04:10 | 2.1 |
        | 2013-05-20 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b
        '''
        _mt._get_metric_tracker().track('timeseries.tshift')
        if type(delta) not in [_datetime.timedelta]:
            raise TypeError("delta must be from type datetime.timedelta.")

        delta_total_seconds = delta.total_seconds()
        tshifted_timeseries = self._timeseries.tshift(delta_total_seconds)
        return TimeSeries(data=None, index=self.index_col_name,
                          gl_timeseries=tshifted_timeseries)

    def shift(self,steps):
        '''
        Shift the non-index columns in the TimeSeries object by specified
        number of steps. The rows at the boundary with no values anymore are
        replaced by None values.

        Parameters
        -----------
        steps : int
            The number of steps to move, positive or negative.

        Returns
        --------
        out : TimeSeries

        Examples
        --------
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                 'b':[t0, t0 + dt.timedelta(days=5),
        ...                      t0 + dt.timedelta(days=10)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> print ts
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        >>> print ts.shift(2)
        +---------------------+------+
        |          b          |  a   |
        +---------------------+------+
        | 2013-05-07 10:04:10 | None |
        | 2013-05-12 10:04:10 | None |
        | 2013-05-17 10:04:10 | 1.1  |
        +---------------------+------+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b
        '''
        _mt._get_metric_tracker().track('timeseries.shift')
        if not isinstance(steps,int):
            raise TypeError("steps must be an integer type.")

        if (steps < 0 and (-1 * steps) > len(self)) or (steps > 0 and steps > len(self)):
            steps = len(self)

        if steps == 0 or len(self.value_col_names) == 0:
            return self

        shifted_timeseries = self._timeseries.shift(steps)
        return TimeSeries(data=None, index=self.index_col_name,
                          gl_timeseries=shifted_timeseries)

    def slice(self, start_time, end_time, closed='left'):
        """
        Returns a new TimeSeries with the range specified by `start_time` and
        `end_time`. Both start and end parameters must be given.

        Parameters
        ----------
        start_time : datetime.datetime
            The beginning of the interval in the returned range. The new
            TimeSeries will have no rows with datetimes earlier than this
            datetime.

        end_time : datetime.datetime
            The end of the interval in the returned range. The new TimeSeries
            will have no rows with datetimes later than this datetime.

        closed : ['left', 'right', 'both', 'neither'], optional
            Determines what sides of the interval are closed (include the given time)

            * left: Only the left side of the range (the start) will be closed

            * right: Only the right side of the range (the end) will be closed

            * both: Both sides of the range will be closed

            * neither: Neither sides of the range will be closed

        Returns
        -------
        out : TimeSeries

        Examples
        --------
        >>> import datetime as dt
        >>> start = dt.datetime(2013, 5, 7)
        >>> end = dt.datetime(2013, 5, 9, 23, 59, 59)
        >>> sa = gl.SArray.date_range(start, end, dt.timedelta(hours=12))
        >>> sf = gl.SFrame({'time': sa,
        ...                 'data': [i for i in range(0, len(sa))]})
        >>> ts = gl.TimeSeries(sf, index='time')
        >>> print ts
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 00:00:00 |  0   |
        | 2013-05-07 12:00:00 |  1   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-08 12:00:00 |  3   |
        | 2013-05-09 00:00:00 |  4   |
        | 2013-05-09 12:00:00 |  5   |
        +---------------------+------+
        [6 rows x 2 columns]
        The index column of the TimeSeries is: time

        >>> ts.slice(dt.datetime(2013,5,7,0,0,0),
        ...                   dt.datetime(2013,5,9,0,0,0))
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 00:00:00 |  0   |
        | 2013-05-07 12:00:00 |  1   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-08 12:00:00 |  3   |
        +---------------------+------+
        [4 rows x 2 columns]
        The index column of the TimeSeries is: time

        >>> ts.slice(dt.datetime(2013,5,7,0,0,0),
        ....                  dt.datetime(2013,5,9,0,0,0), closed='both')
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 00:00:00 |  0   |
        | 2013-05-07 12:00:00 |  1   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-08 12:00:00 |  3   |
        | 2013-05-09 00:00:00 |  4   |
        +---------------------+------+
        [5 rows x 2 columns]
        The index column of the TimeSeries is: time

        >>> ts.slice(dt.datetime(2013,5,7,0,0,0),
        ...                   dt.datetime(2013,5,9,0,0,0), closed='right')
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 12:00:00 |  1   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-08 12:00:00 |  3   |
        | 2013-05-09 00:00:00 |  4   |
        +---------------------+------+
        [4 rows x 2 columns]
        The index column of the TimeSeries is: time

        >>> ts.slice(dt.datetime(2013,5,7,0,0,0),
        ...                   dt.datetime(2013,5,9,0,0,0), closed='neither')
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 12:00:00 |  1   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-08 12:00:00 |  3   |
        +---------------------+------+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: time
        """
        _mt._get_metric_tracker().track('timeseries.slice',
                                        properties={'closed':closed})
        if closed not in ['left','right','both','neither']:
            raise TypeError("Parameter 'closed' must be one of ['left', 'right', 'both', 'neither']")

        if not isinstance(start_time, _datetime.datetime):
            raise TypeError("Parameter 'start_time' must be datetime.datetime")

        if not isinstance(end_time, _datetime.datetime):
            raise TypeError("Parameter 'end_time' must be datetime.datetime")

        ret_ts = self._timeseries.slice(start_time, end_time, closed)
        return TimeSeries(data=None, index=self.index_col_name,
                          gl_timeseries=ret_ts)

    @property
    def min_time(self):
        """
        The minimum value of the time index.

        Returns
        -------
        out : datetime.datetime
        """
        if len(self) < 1:
            return None
        return self._sframe[self.index_col_name][0]

    @property
    def max_time(self):
        """
        The maximum value of the time index.

        Returns
        -------
        out : datetime.datetime
        """
        if len(self) < 1:
            return None
        return self._sframe[self.index_col_name][-1]

    @property
    def range(self):
        """
        The minimum and maximum value of the time index.

        Returns
        -------
        out : tuple
        """
        return (self.min_time,self.max_time)

    def _is_date_or_time(self, key):
        return isinstance(key,_datetime.date) or isinstance(key,_datetime.time)

    def __setitem__(self, key, value):
        """
        A wrapper around add_column.  Key is str.  If value is an SArray, it is
        added to the TimeSeries as a column.  If it is a constant value (int,
        str, or float), then a column is created where every entry is equal to
        the constant value.  Existing columns cannot be replaced using this
        wrapper.
        """
        if type(key) is str:
            if key == self.index_col_name:
                raise RuntimeError('Cannot reassign index column %s' % self.index_col_name)

            sa_value = None
            if (type(value) is _graphlab.SArray):
                sa_value = value
            elif hasattr(value, '__iter__') and not(_graphlab.util._is_string(value)):  
                # wrap list, array... to sarray
                sa_value = _graphlab.SArray(value)
            else:  # create an sarray  of constant value
                sa_value = _graphlab.SArray.from_const(value, self.__len__())

            # set new column
            if not key in self.column_names():
                self.add_column(sa_value, key)
            else:
                tmpname = '__' + '-'.join(self.column_names())
                try:
                    self.add_column(sa_value, tmpname)
                    self.swap_columns(key, tmpname)
                    self.remove_column(key)
                    self.rename({tmpname: key})
                except Exception as e:
                    raise
        else:
            raise TypeError('Cannot set column with key type ' + str(type(key)))

    def __delitem__(self, key):
        """
        Wrapper around remove_column.
        """
        self.remove_column(key)

    def __getitem__(self, key):
        """
        This method does things based on the type of `key`.

        If `key` is:
            * datetime.datetime
                Returns a TimeSeries with rows only with this exact datetime in
                the time index
            * datetime.date
                Returns a TimeSeries with rows only occurring on the given date
            * str
                selects column with name 'key'
            * type
                selects all columns with types matching the type
            * list of str or type
                selects all columns with names or type in the list
            * SArray
                Performs a logical filter.  Expects given SArray to be the same
                length as all columns in current SFrame.  Every row
                corresponding with an entry in the given SArray that is
                equivalent to False is filtered from the result. Will return
                result as TimeSeries if possible.
            * int
                Returns a single row of the TimeSeries (the `key`th one) as a
                dictionary.
            * slice
                Returns a TimeSeries including only the sliced rows. When a
                datetime.datetime is given, the time index is used to determine
                the slice
        """
        # Investigates any possible way to do a timeseries-specific selection.
        # If nothing in this if block returns something, we fall back to
        # SFrame's getitem method
        if type(key) is slice:
            start = key.start
            stop = key.stop
            step = key.step
            valid_date_range = False
            closed = 'left'
            ret_ts = None

            # A slice with two ends, both datetime.datetime
            if (start is not None) and (stop is not None):
                if isinstance(start, _datetime.datetime) and isinstance(stop,
                        _datetime.datetime):
                    valid_date_range = True
            # An open-ended slice with datetime.datetime
            elif isinstance(start,_datetime.datetime) or isinstance(stop,
                    _datetime.datetime):
                if start is None:
                    start = self.min_time
                if stop is None:
                    stop = self.max_time
                    closed = 'both'
                valid_date_range = True

            if valid_date_range:
                ret_ts = self.slice(start, stop, closed=closed)
                if step is not None and step > 1:
                    return TimeSeries(ret_ts._sframe[::step],
                                      index=ret_ts.index_col_name,
                                      is_sorted=True)
                else:
                    return ret_ts
            # If no valid date range was found, but dates were given, we
            # shouldn't fall back to the confusing error from SFrame's getitem
            elif self._is_date_or_time(start) or self._is_date_or_time(stop):
                raise TypeError("Slices only accept datetime.datetime.")
        elif self._is_date_or_time(key):
            index_col = self._sframe[self.index_col_name]
            t1 = None
            t2 = None
            if not hasattr(key,"hour"):
                t1 = _datetime.datetime.combine(key, _datetime.datetime.min.time())
                t2 = t1 + _datetime.timedelta(days=1)
                return self.slice(t1,t2)
            elif not hasattr(key,"day"):
                # datetime.time. Not supported for now.
                pass
            else:
                return TimeSeries(data=self._sframe[index_col == key],
                                  index=self.index_col_name,
                                  is_sorted=True)

        # Fall back to SFrame getitem
        gi_ret = self._sframe.__getitem__(key)

        # This is meant to keep the user from needing to convert an SFrame back
        # to TimeSeries. In case ``gi_ret`` is an SFrame, we automatically add
        # the TimeSeries index column to ``gi_ret`` (if it is not already
        # threre) and convert it to a TimeSeries object. The idea is that
        # TimeSeries.__getitem__() should not return
        # an SFrame, instead a TimeSeries object. Note that no operations on
        # SFrame's getitem will shuffle the order of the SFrame (even if it
        # did, the TimeSeries constructor would re-sort)
        if gi_ret.__class__.__name__ == 'SFrame':
            cnames = gi_ret.column_names()
            if not self.index_col_name in cnames:
                gi_ret.add_column(self[self.index_col_name],
                        name=self.index_col_name)
            try:
                return TimeSeries(gi_ret, index=self.index_col_name,
                        is_sorted=True)
            except:
                pass
        return gi_ret

    def to_sframe(self, include_index = True):
        '''
        Convert the TimeSeries to an SFrame. (zero copy)

        Parameters
        ----------
        include_index : bool (default True)
            A flag to determine if the index must be returned or not.

        Returns
        -------
        out : SFrame

        Examples
        --------
        .. sourcecode:: python

          >>> import graphlab as gl
          >>> import datetime as dt

          >>> index = gl.SArray.date_range(start_time = dt.datetime(2015, 1, 1),
          ...                              end_time = dt.datetime(2016, 1, 1),
          ...                              freq = dt.timedelta(days = 1))
          >>> value = gl.SArray(range(len(index)))

          >>> data = gl.SFrame({'index': index, 'value': value})
          >>> ts = gl.TimeSeries(data, index='index')

          # The TimeSeries object can be converted to an SFrame as follows.
          >>> data = ts.to_sframe()
          >>> print data
          +---------------------+-------+
          |        index        | value |
          +---------------------+-------+
          | 2015-01-01 00:00:00 |   0   |
          | 2015-01-02 00:00:00 |   1   |
          | 2015-01-03 00:00:00 |   2   |
          | 2015-01-04 00:00:00 |   3   |
          | 2015-01-05 00:00:00 |   4   |
          | 2015-01-06 00:00:00 |   5   |
          | 2015-01-07 00:00:00 |   6   |
          | 2015-01-08 00:00:00 |   7   |
          | 2015-01-09 00:00:00 |   8   |
          | 2015-01-10 00:00:00 |   9   |
          +---------------------+-------+
          [366 rows x 2 columns]

          # Sometimes, it is useful to drop the index column, which can also be
          # done:
          >>> data = ts.to_sframe(include_index = False)
          >>> print data
          Data:
          +-------+
          | value |
          +-------+
          |   0   |
          |   1   |
          |   2   |
          |   3   |
          |   4   |
          |   5   |
          |   6   |
          |   7   |
          |   8   |
          |   9   |
          +-------+
          [366 rows x 1 columns]
        '''
        _mt._get_metric_tracker().track('timeseries.to_sframe')
        if include_index:
          return self._timeseries.sframe
        else:
          return self._timeseries.sframe.remove_column(self.index_col_name)

    def head(self, n=10):
        '''
        The first ``n`` rows of the TimeSeries.
        See :py:func:`~graphlab.SFrame.head` for documentation.
        '''
        return self._sframe.head(n)

    def tail(self, n=10):
        '''
        The last ``n`` rows of the TimeSeries.
        See :py:func:`~graphlab.SFrame.tail` for documentation.
        '''
        return self._sframe.tail(n)

    def union(self, other):
        '''
        Union the TimeSeries object with the `other` TimeSeries object. If the
        TimeSeries.range() of two TimeSeries does not overlap, then this
        operation is basically appending two TimeSeries. The output TimeSeries
        borrows the schema (column types and column names) of the caller
        TimeSeries.

        Parameters
        ----------
        other : TimeSeries
            The other TimeSeries object to union with.

        Returns
        -------
        out : TimeSeries

        Examples
        --------
        >>> import graphlab as gl
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf1 = gl.SFrame({'a1': [1.1, 2.1, 3.1],
        ...                  'index1': [t0, t0 + dt.timedelta(days=3),
        ...                             t0 + dt.timedelta(days=6)]})
        >>> ts1 = gl.TimeSeries(sf1, index='index1')
        >>> print ts1
        +---------------------+-----+
        |        index1       |  a1 |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-10 10:04:10 | 2.1 |
        | 2013-05-13 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: index1

        >>> sf2 = gl.SFrame({'a1': [5.1, 6.1, 7.1],
        ...                  'index1': [t0 + dt.timedelta(days=1),
        ...                             t0 + dt.timedelta(days=4),
        ...                             t0 + dt.timedelta(days=8)]})
        >>> ts2 = gl.TimeSeries(sf2, index='index1')
        >>> print ts2
        +---------------------+-----+
        |        index1       |  a1 |
        +---------------------+-----+
        | 2013-05-08 10:04:10 | 5.1 |
        | 2013-05-11 10:04:10 | 6.1 |
        | 2013-05-15 10:04:10 | 7.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: index2

        >>> ts_union = ts1.union(ts2)
        >>> print ts_union
        +---------------------+-----+
        |        index1       |  a1 |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-08 10:04:10 | 5.1 |
        | 2013-05-10 10:04:10 | 2.1 |
        | 2013-05-11 10:04:10 | 6.1 |
        | 2013-05-13 10:04:10 | 3.1 |
        | 2013-05-15 10:04:10 | 7.1 |
        +---------------------+-----+
        [6 rows x 2 columns]
        The index column of the TimeSeries is: index1
        '''
        _mt._get_metric_tracker().track('timeseries.union')
        if not isinstance(other,TimeSeries):
            raise TypeError("The `other` argument must be a TimeSeries object.")

        union_timeseries = self._timeseries.ts_union(other._timeseries)
        return TimeSeries(data=None, index=self.index_col_name,
                          gl_timeseries=union_timeseries)

    def index_join(self, other, how='inner', index_col_name=None):
        '''
        Join the TimeSeries object with the `other` TimeSeries object based on
        the join method. TimeSeries.index_join() supports joining two
        timeseries on their index column and outputs a new TimeSeries.

        There is another method called TimeSeries.join() which joins the
        TimeSeries with an SFrame using a SQL-style equi-join operation by
        columns and generates as output a new SFrame.

        Parameters
        ----------
        other : TimeSeries
            The other TimeSeries object to join with.

        how : {'inner', 'left', 'right', 'outer'}, optional
            Join method. Default value is 'inner'.

        index_col_name : str, optional
            The index column name for the output TimeSeries.

        Returns
        -------
        out : TimeSeries

        Examples
        --------
        >>> import graphlab as gl
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf1 = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                  'b': [t0, t0 + dt.timedelta(days=1),
        ...                        t0 + dt.timedelta(days=2)]})
        >>> ts1 = gl.TimeSeries(sf1, index='b')
        >>> print ts1
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-08 10:04:10 | 2.1 |
        | 2013-05-09 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

        >>> sf2 = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                  'b': [t0 + dt.timedelta(days=1),
        ...                        t0 + dt.timedelta(days=2),
        ...                        t0 + dt.timedelta(days=3)]})
        >>> ts2 = gl.TimeSeries(sf2, index='b')
        >>> print ts2
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-08 10:04:10 | 1.1 |
        | 2013-05-09 10:04:10 | 2.1 |
        | 2013-05-10 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b

        >>> ts_join = ts1.index_join(ts2, how='inner')
        >>> print ts_join
        +---------------------+-----+-----+
        |          b          |  a  | a.1 |
        +---------------------+-----+-----+
        | 2013-05-08 10:04:10 | 2.1 | 1.1 |
        | 2013-05-09 10:04:10 | 3.1 | 2.1 |
        +---------------------+-----+-----+
        [2 rows x 3 columns]
        The index column of the TimeSeries is: b

        >>> ts_join = ts1.index_join(ts2, how='outer')
        >>> print ts_join
        +---------------------+------+------+
        |          b          |  a   | a.1  |
        +---------------------+------+------+
        | 2013-05-07 10:04:10 | 1.1  | None |
        | 2013-05-08 10:04:10 | 2.1  | 1.1  |
        | 2013-05-09 10:04:10 | 3.1  | 2.1  |
        | 2013-05-10 10:04:10 | None | 3.1  |
        +---------------------+------+------+
        [4 rows x 3 columns]
        The index column of the TimeSeries is: b

        >>> ts_join = ts1.index_join(ts2, how='left')
        >>> print ts_join
        +---------------------+-----+------+
        |          b          |  a  | a.1  |
        +---------------------+-----+------+
        | 2013-05-07 10:04:10 | 1.1 | None |
        | 2013-05-08 10:04:10 | 2.1 | 1.1  |
        | 2013-05-09 10:04:10 | 3.1 | 2.1  |
        +---------------------+-----+------+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: b

        >>> ts_join = ts1.index_join(ts2, how='right')
        >>> print ts_join
        +---------------------+------+-----+
        |          b          |  a   | a.1 |
        +---------------------+------+-----+
        | 2013-05-08 10:04:10 | 2.1  | 1.1 |
        | 2013-05-09 10:04:10 | 3.1  | 2.1 |
        | 2013-05-10 10:04:10 | None | 3.1 |
        +---------------------+------+-----+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: b
        '''
        _mt._get_metric_tracker().track('timeseries.index_join',
                                        properties={'how':how})
        if not isinstance(other,TimeSeries):
            raise TypeError("The `other` argument must be a TimeSeries object.")
        if how not in ['inner','left','right','outer']:
            raise ValueError("The `how` argument must be selected from: {'inner', 'left', 'right', 'outer'}")
        if index_col_name is None:
            index_col_name = self.index_col_name
        if not isinstance(index_col_name,str):
            raise TypeError("The `index_col_name` argument must be str")

        joined_timeseries = self._timeseries.index_join(other._timeseries, how,
                                                        index_col_name)
        return TimeSeries(data=None, index=index_col_name,
                          gl_timeseries=joined_timeseries)

    def join(self, other, on=None, how='inner'):
        '''
        Join the current (left) TimeSeries with the given (right) SFrame using
        a SQL-style equi-join operation by column and generates as output a new
        SFrame.

        There is another method called TimeSeries.index_join() that supports
        joining two timeseries on their index column and outputs a new
        TimeSeries.

        Parameters
        ----------
        other : SFrame
            The SFrame to join.

        on : None | str | list | dict, optional
            The column name(s) representing the set of join keys.  Each row that
            has the same value in this set of columns will be merged together.

            * If 'None' is given, join will use all columns that have the same
              name as the set of join keys.

            * If a str is given, this is interpreted as a join using one column,
              where both SFrame and TimeSeries have the same column name.

            * If a list is given, this is interpreted as a join using one or
              more column names, where each column name given exists in both
              SFrame and TimeSeries.

            * If a dict is given, each dict key is taken as a column name in the
              TimeSeries, and each dict value is taken as the column name in
              right SFrame that will be joined together. e.g.
              {'left_col_name':'right_col_name'}.

        how : {'inner', 'left', 'right', 'outer'}, optional
            The type of join to perform. 'inner' is default.

            * inner: Equivalent to a SQL inner join.  Result consists of the
              rows from both TimeSeries and SFrame whose join key values match
              exactly, merged together into one SFrame.

            * left: Equivalent to a SQL left outer join. Result is the union
              between the result of an inner join and the rest of the rows from
              the TimeSeries, merged with missing values.

            * right: Equivalent to a SQL right outer join.  Result is the union
              between the result of an inner join and the rest of the rows from
              the right SFrame, merged with missing values.

            * outer: Equivalent to a SQL full outer join. Result is
              the union between the result of a left outer join and a right
              outer join.

        Returns
        -------
        out : SFrame

        Notes
        ------
        See :py:func:`~graphlab.SFrame.join` for examples.
        '''
        _mt._get_metric_tracker().track('timeseries.join',
                                        properties={'type':how})
        if not isinstance(other,_graphlab.SFrame):
            raise TypeError("The `other` argument must be an SFrame object.")
        return self._timeseries.sframe.join(other,on,how)

    def filter_by(self, values, column_name, exclude=False):
        """
        Filter a TimeSeries by values inside an iterable object. Result is an
        TimeSeries that only includes (or excludes) the rows that have a column
        with the given ``column_name`` which holds one of the values in the
        given ``values`` :class:`~graphlab.SArray`. If ``values`` is not an
        SArray, we attempt to convert it to one before filtering.

        Parameters
        ----------
        values : SArray | list | numpy.ndarray | pandas.Series | str
            The values to use to filter the SFrame.  The resulting SFrame will
            only include rows that have one of these values in the given
            column.

        column_name : str
            The column of the SFrame to match with the given `values`.

        exclude : bool
            If True, the result SFrame will contain all rows EXCEPT those that
            have one of ``values`` in ``column_name``.

        Returns
        -------
        out : TimeSeries
            The filtered TimeSeries.

        Examples
        --------
        See :py:func:`~graphlab.SFrame.filter_by` for examples.
        """
        _mt._get_metric_tracker().track('timeseries.filter_by')
        do_index_join = False
        index_join_fake_col_name = ""
        if type(column_name) is not str:
            raise TypeError("Must pass a str as column_name")

        existing_columns = self.column_names()
        if column_name not in existing_columns:
            raise KeyError("Column '" + column_name + "' not in TimeSeries.")

        if type(values) is not _graphlab.SArray:
            # If we were given a single element, try to put in list and convert
            # to SArray
            if not _is_non_string_iterable(values):
                values = [values]
            values = _graphlab.SArray(values)

        value_sf = _graphlab.SFrame()
        value_sf.add_column(values, column_name)

        existing_type = self.column_types()[self.column_names().index(column_name)]
        given_type = value_sf.column_types()[0]
        if given_type != existing_type:
            raise TypeError("Type of given values does not match type of column '" +
                column_name + "' in TimeSeries.")

        # Make sure the values list has unique values, or else join will not
        # filter.
        value_sf = value_sf.groupby(column_name, {})

        # Check if index_join is better (sort-merge join vs. hash join)
        if (column_name == self.index_col_name) and (given_type == _datetime.datetime):
            # Rough check to see if sorting this will be worth the trouble
            do_index_join = True;
            # TODO: Fake column to let this become a TimeSeries
            index_join_fake_col_name = column_name + "foo"
            value_sf[index_join_fake_col_name] = 0

        if exclude:
            id_name = "id"
            # Make sure this name is unique so we know what to remove in
            # the result
            while id_name in existing_columns:
                id_name += "1"
            value_sf = value_sf.add_row_number(id_name)

            tmp = None
            if do_index_join:
                tmp = self.index_join(TimeSeries(value_sf, column_name), 'left')
                tmp.remove_column(tmp._sframe.column_names()[-1])
            else:
                tmp = self.join(value_sf, how='left', on={column_name:column_name})
                tmp = TimeSeries(data=tmp, index=self.index_col_name)

            ret = tmp[tmp[id_name] == None]
            ret.remove_column(id_name)
            return ret
        else:
            if do_index_join:
                ret = self.index_join(TimeSeries(value_sf, column_name), 'inner')
                ret.remove_column(ret._sframe.column_names()[-1])
                return ret
            tmp = self.join(value_sf, how='inner', on={column_name:column_name})
            return TimeSeries(data=tmp, index=self.index_col_name)

    def dropna(self, columns=None, how='any'):
        """
        See :py:func:`~graphlab.SFrame.dropna` for documentation.
        """
        _mt._get_metric_tracker().track('timeseries.dropna',
                                        properties={'how': how})
        sf = self._sframe.dropna(columns, how)
        return TimeSeries(data=sf, index=self.index_col_name, is_sorted=True)

    def dropna_split(self, columns=None, how='any'):
        """
        See :py:func:`~graphlab.SFrame.dropna_split` for documentation.
        """
        _mt._get_metric_tracker().track('timeseries.dropna_split',
                                        properties={'how': how})
        sf = self._sframe.dropna_split(columns, how)
        return (TimeSeries(data=sf[0], index=self.index_col_name, is_sorted=True),
                TimeSeries(data=sf[1], index=self.index_col_name, is_sorted=True))

    def save(self, location):
        '''
        Save the TimeSeries object to the given location.

        Parameters
        ----------
        location: str
            The path to save the TimeSeries object.

        See Also
        --------
        graphlab.load_timeseries

        Examples
        --------
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                 'b': [t0, t0 + dt.timedelta(days=5),
        ...                       t0 + dt.timedelta(days=10)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> ts.save("my_series")
        >>> ts2 = gl.load_timeseries("my_series")
        >>> print ts2
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b
        '''
        _mt._get_metric_tracker().track('timeseries.save')
        if not isinstance(location,str):
            raise TypeError("The `location` argument must be from type str.")
        _glconnect.get_unity().save_model(self._timeseries,
                _make_internal_url(location))

    def group(self, key_columns):
        """
        Separate a TimeSeries by the distinct values in one or more columns.

        The output is a `graphlab.timeseries.GroupedTimeSeries` object, which
        provides an interface for retrieving one or more groups by their group
        name, or iterating through all groups. More information on this
        interface can be found at `graphlab.timeseries.GroupedTimeSeries`.

        Each group is a separate TimeSeries, which possesses the same columns
        as the original TimeSeries.

        To group the TimeSeries by a part of it's timestamp (e.g. "DAY" or
        "HOUR"), use the special types declared in
        `graphlab.TimeSeries.date_part`.

        Parameters
        ----------
        key_columns : str | list
            The columns to group by. Can be a single column name or a list of
            column names.

        Returns
        -------
        out : GroupedTimeSeries

        Examples
        --------
        >>> import datetime as dt
        >>> start = dt.datetime(2013, 5, 7)
        >>> end = dt.datetime(2013, 5, 9, 23, 59, 59)
        >>> sa = gl.SArray.date_range(start, end, dt.timedelta(hours=12))
        >>> sf = gl.SFrame({'time': sa,
        ...                 'data': [i for i in range(0, len(sa))]})
        >>> ts = gl.TimeSeries(sf, index='time')
        >>> print ts
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 00:00:00 |  0   |
        | 2013-05-07 12:00:00 |  1   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-08 12:00:00 |  3   |
        | 2013-05-09 00:00:00 |  4   |
        | 2013-05-09 12:00:00 |  5   |
        +---------------------+------+
        [6 rows x 2 columns]
        The index column of the TimeSeries is: time

        >>> by_hour = ts.group(ts.date_part.HOUR)
        >>> by_hour.groups()
        dtype: int
        Rows: 2
        [0, 12]

        >>> by_hour.get_group(0)
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 00:00:00 |  0   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-09 00:00:00 |  4   |
        +---------------------+------+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: time

        >>> by_hour.get_group(12)
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 12:00:00 |  1   |
        | 2013-05-08 12:00:00 |  3   |
        | 2013-05-09 12:00:00 |  5   |
        +---------------------+------+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: time

        >>> for name, group in by_hour:
        ...     print "Group name: " + str(name)
        ...     print group
        ...
        Group name: 0
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 00:00:00 |  0   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-09 00:00:00 |  4   |
        +---------------------+------+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: time
        Group name: 12
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 12:00:00 |  1   |
        | 2013-05-08 12:00:00 |  3   |
        | 2013-05-09 12:00:00 |  5   |
        +---------------------+------+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: time

        >>> by_day = ts.group([ts.date_part.YEAR, ts.date_part.MONTH,
        ...                    ts.date_part.DAY])
        >>> by_day.groups()
        dtype: list
        Rows: 3
        [[2013, 5, 7], [2013, 5, 8], [2013, 5, 9]]

        >>> by_day.get_group([2013,5,7])
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 00:00:00 |  0   |
        | 2013-05-07 12:00:00 |  1   |
        +---------------------+------+
        [2 rows x 2 columns]
        The index column of the TimeSeries is: time
        """
        _mt._get_metric_tracker().track('timeseries.group')
        gts = GroupedTimeSeries(self, key_columns)
        return gts

    def add_column(self, data, name=""):
        """
        Add a column to this TimeSeries object. The number of elements in the
        data given must match the length of every other column of the
        TimeSeries. This operation modifies the current TimeSeries in place and
        returns self. If no name is given, a default name is chosen.

        Parameters
        ----------
        data : SArray
            The 'column' of data to add.

        name : string, optional
            The name of the column. If no name is given, a default name is
            chosen.

        Returns
        -------
        out : TimeSeries
            The current TimeSeries

        See Also
        --------
        TimeSeries.remove_column

        Examples
        --------
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                 'b': [t0, t0 + dt.timedelta(days=5),
        ...                       t0 + dt.timedelta(days=10)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> ts.add_column(gl.SArray([1,2,3]), "new_value")
        >>> print ts
        +---------------------+-----+-----------+
        |          b          |  a  | new_value |
        +---------------------+-----+-----------+
        | 2013-05-07 10:04:10 | 1.1 |     1     |
        | 2013-05-12 10:04:10 | 2.1 |     2     |
        | 2013-05-17 10:04:10 | 3.1 |     3     |
        +---------------------+-----+-----------+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: b

        >>> ts.remove_column("new_value")
        >>> print ts
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b
        """
        if not isinstance(data, _graphlab.SArray):
            raise TypeError("Must give column as SArray")
        if not isinstance(name, str):
            raise TypeError("Invalid column name: must be str")

        self._timeseries.add_column(data,name)
        self.value_col_names = self._timeseries.value_col_names
        self._sframe = self._timeseries.sframe
        return self

    def swap_columns(self, column_1, column_2):
        """
        Swap the columns with the given names. This operation modifies the
        current TimeSeries in place and returns self. `name` cannot be the name
        of the index column.

        Parameters
        ----------
        column_1 : string
            Name of column to swap

        column_2 : string
            Name of other column to swap

        Returns
        -------
        out : TimeSeries
            The TimeSeries with swapped columns.

        Examples
        --------
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a0': [1.1, 2.1, 3.1],
        ...                 'a1': [2.2, 3.1, 4.1],
        ...                 'b': [t0, t0 + dt.timedelta(days=5),
        ...                       t0 + dt.timedelta(days=10)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> ts
        +---------------------+-----+-----+
        |          b          |  a0 |  a1 |
        +---------------------+-----+-----+
        | 2013-05-07 10:04:10 | 1.1 | 2.2 |
        | 2013-05-12 10:04:10 | 2.1 | 3.1 |
        | 2013-05-17 10:04:10 | 3.1 | 4.1 |
        +---------------------+-----+-----+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: b

        >>> ts.swap_columns('a0', 'a1')
        >>> ts
        +---------------------+-----+-----+
        |          b          |  a1 |  a0 |
        +---------------------+-----+-----+
        | 2013-05-07 10:04:10 | 2.2 | 1.1 |
        | 2013-05-12 10:04:10 | 3.1 | 2.1 |
        | 2013-05-17 10:04:10 | 4.1 | 3.1 |
        +---------------------+-----+-----+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: b
        """
        if self.index_col_name in [column_1, column_2]:
            raise ValueError('Cannot swap index column %s.' % self.index_col_name)
        for item in [column_1,column_2]:
            if item not in self.value_col_names:
                raise ValueError('Cannot find column %s.' % item)
        self._sframe.swap_columns(column_1, column_2)
        self._timeseries.sframe = self._sframe
        self.value_col_names = [val for val in self._sframe.column_names() if val != self.index_col_name]
        self._timeseries.value_col_names = self.value_col_names

    def remove_column(self, name):
        """
        Remove a column from this TimeSeries. This operation modifies the
        current TimeSeries in place and returns self. `name` cannot be the name
        of the index column.

        Parameters
        ----------
        name : string
            The name of the column to remove.

        Returns
        -------
        out : TimeSeries
            The TimeSeries with given column removed.

        See Also
        ---------
        TimeSeries.add_column

        Examples
        --------
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                 'b': [t0, t0 + dt.timedelta(days=5),
        ...                       t0 + dt.timedelta(days=10)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> ts.add_column(gl.SArray([1,2,3]), "new_value")
        >>> print ts
        +---------------------+-----+-----------+
        |          b          |  a  | new_value |
        +---------------------+-----+-----------+
        | 2013-05-07 10:04:10 | 1.1 |     1     |
        | 2013-05-12 10:04:10 | 2.1 |     2     |
        | 2013-05-17 10:04:10 | 3.1 |     3     |
        +---------------------+-----+-----------+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: b

        >>> ts.remove_column("new_value")
        >>> print ts
        +---------------------+-----+
        |          b          |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: b
        """
        name = str(name)
        if name == self.index_col_name:
            raise ValueError('Cannot remove index column %s' % self.index_col_name)
        if name not in self.value_col_names:
            raise ValueError('Cannot find column %s' % name)
        self._timeseries.remove_column(name)
        self.value_col_names = self._timeseries.value_col_names
        self._sframe = self._timeseries.sframe

        return self

    def rename(self, names):
        """
        Rename the given columns. `names` is expected to be a dict specifying
        the old and new names. This changes the names of the columns given as
        the keys and replaces them with the names given as the values.  This
        operation modifies the current TimeSeries in place and returns self.

        Parameters
        ----------
        names : dict [string, string]
            Dictionary of [old_name, new_name]

        Returns
        -------
        out : TimeSeries
            The current TimeSeries.

        Examples
        --------
        >>> import datetime as dt
        >>> t0 = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> sf = gl.SFrame({'a': [1.1, 2.1, 3.1],
        ...                 'b': [t0, t0 + dt.timedelta(days=5),
        ...                       t0 + dt.timedelta(days=10)]})
        >>> ts = gl.TimeSeries(sf, index='b')
        >>> ts.rename({'b': 'index'})
        >>> print ts
        +---------------------+-----+
        |        index        |  a  |
        +---------------------+-----+
        | 2013-05-07 10:04:10 | 1.1 |
        | 2013-05-12 10:04:10 | 2.1 |
        | 2013-05-17 10:04:10 | 3.1 |
        +---------------------+-----+
        [3 rows x 2 columns]
        The index column of the TimeSeries is: index
        """
        if (type(names) is not dict):
            raise TypeError('names must be a dictionary: oldname -> newname')
        all_columns = set(self.to_sframe().column_names())
        for k in names:
            if not k in all_columns:
                raise ValueError('Cannot find column %s in the TimeSeries' % k)
            if k == self.index_col_name:
                self.index_col_name = names[k]

        self._timeseries.sframe = self._timeseries.sframe.rename(names)
        self._sframe = self._timeseries.sframe

        self.value_col_names = []
        new_all_columns = set(self.to_sframe().column_names())
        for k in new_all_columns:
            if k != self.index_col_name:
                self.value_col_names.append(k)

        self._timeseries.index_col_name = self.index_col_name
        self._timeseries.value_col_names = self.value_col_names

        return self

    def apply(self, fn, dtype=None, seed=None):
        '''
        See :py:func:`~graphlab.SFrame.apply` for documentation.
        '''
        return self.to_sframe().apply(fn,dtype,seed)
