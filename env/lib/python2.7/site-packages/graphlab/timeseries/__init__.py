"""
The TimeSeries object is the fundamental data structure for multivariate time
series data. TimeSeries objects are backed by a single `SFrame`, but include
extra metadata.

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

Each column in the table is a univariate time series, and the index is shared
across all of the series.

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
"""
