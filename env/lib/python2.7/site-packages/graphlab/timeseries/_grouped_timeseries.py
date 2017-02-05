"""
An interface for accessing an TimeSeries that is grouped by the values it
contains in one or more columns.
"""

'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''

from .. import connect as _mt
import graphlab as _graphlab

class GroupedTimeSeries(object):
    def __init__(self, data, key_columns):
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
        data : Timeseries
            The single time series (before grouping).

        key_columns : str | list
            The columns to group by. Can be a single column name or a list of
            column names.

        Returns
        -------
        out : GroupedTimeSeries

        Examples
        --------

        # Create some sample data.
        >>> import datetime as dt
        >>> start = dt.datetime(2013, 5, 7)
        >>> end = dt.datetime(2013, 5, 9, 23, 59, 59)
        >>> sa = gl.SArray.date_range(start, end, dt.timedelta(hours=12))
        >>> sf = gl.SFrame({'time': sa,
        ...                 'data': [i for i in range(0, len(sa))]})


        # Create a time series from data.
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

        # Compute the number of groups.
        >>> by_hour = ts.group(ts.date_part.HOUR)
        >>> by_hour.groups()
        dtype: int
        Rows: 2
        [0, 12]

        # Get the group for the first hour.
        >>> by_hour.get_group(0)
        +---------------------+------+
        |         time        | data |
        +---------------------+------+
        | 2013-05-07 00:00:00 |  0   |
        | 2013-05-08 00:00:00 |  2   |
        | 2013-05-09 00:00:00 |  4   |
        +---------------------+------+

        """
        _mt._get_metric_tracker().track('grouped_timeseries.__init__')
        self._date_parts = data.date_part.__dict__.keys()
        self._grouped_ts = _graphlab.extensions._GroupedTimeseries()
        self.index_col_name = data.index_col_name

        if isinstance(key_columns, str) or self._is_date_part(key_columns):
            key_columns = [key_columns]

        if not isinstance(key_columns, list):
            raise TypeError("Input `key_columns` must be of type str or list.")

        sf = data.to_sframe()

        # Pass this to split_datetime
        split_limit = []
        count = 0
        ids_of_date_parts = []
        for key in key_columns:
            if self._is_date_part(key):
                split_limit.append(key.name)
                ids_of_date_parts.append(count)
            count += 1


        # This is a bit of a hack to support grouping by "date parts". We split
        # the datetime found in the time index by the given date parts, so we
        # have a separate column for each. Then we don't need special logic in
        # the actual group algorithm for this special case. We then remove
        # these "temp columns" whenever we need to return the result back to
        # the user.
        self._temp_col_names = []
        if len(split_limit) > 0:
            temp_date_cols = sf[[self.index_col_name]].split_datetime(
                self.index_col_name,limit=split_limit)
            # Assumes split_datetime always returns rows in the same order
            num_temp_date_cols = temp_date_cols.num_cols()
            sf.add_columns(temp_date_cols)
            self._temp_col_names = sf.column_names()[-num_temp_date_cols:]

        # In case the column names have changed, add names after added to
        # SFrame to group
        count = 0
        for i in ids_of_date_parts:
            # Replace the classes with the names of the temp columns
            key_columns[i] = self._temp_col_names[count]
            count += 1

        self._grouped_ts.group(sf, self.index_col_name, key_columns)
        self.key_columns = self._grouped_ts.key_columns

    def _is_date_part(self,key):
        """
        Internal method to figure out whether a value comes from the index
        class within graphlab.TimeSeries. These are things like "DAY" and
        "HOUR".
        """
        if hasattr(key, "__class__") and hasattr(key, "__module__"):
            if key.__module__ == 'graphlab.timeseries._timeseries':
                if key.__class__.__name__ in self._date_parts:
                    return True
        return False

    def get_group(self, name):
        """
        Get the TimeSeries associated with the group `name`.

        The name of the group corresponds to the distinct value in the
        column(s) that the group was performed on. Check the output of
        `graphlab.timeseries.GroupedTimeSeries.groups` for all available group
        names.

        Parameters
        ----------
        name : type | list
            Name of the group(s). If more than one column, the name is a list
            of the values of the group, in the same order that they were
            expressed to the group call.

        Returns
        -------
        ts : `graphlab.TimeSeries`

        Examples
        --------
        >>> import datetime as dt
        >>> start = dt.datetime(2013, 5, 7)
        >>> end = dt.datetime(2013, 5, 9, 23, 59, 59)
        >>> sa = gl.TimeSeries.date_range(start,end,dt.timedelta(hours=12))
        >>> sf = gl.SFrame({'time':sa,
        ... 'numbers':[(i % 2) for i in range(0,len(sa))],
        ... 'words':['day' if (i % 2) else 'night' for i in range(0,len(sa))]})

        # Create a timeseries.
        >>> ts = gl.TimeSeries(sf, index='time')
        >>> print ts
        +---------------------+---------+-------+
        |         time        | numbers | words |
        +---------------------+---------+-------+
        | 2013-05-07 00:00:00 |    0    | night |
        | 2013-05-07 12:00:00 |    1    |  day  |
        | 2013-05-08 00:00:00 |    0    | night |
        | 2013-05-08 12:00:00 |    1    |  day  |
        | 2013-05-09 00:00:00 |    0    | night |
        | 2013-05-09 12:00:00 |    1    |  day  |
        +---------------------+---------+-------+
        [6 rows x 3 columns]
        The index column of the TimeSeries is: time

        # Group the timeseries by hour.
        >>> by_hour = ts.group(ts.date_part.HOUR)
        >>> by_hour.get_group(12)
        +---------------------+---------+-------+
        |         time        | numbers | words |
        +---------------------+---------+-------+
        | 2013-05-07 12:00:00 |    1    |  day  |
        | 2013-05-08 12:00:00 |    1    |  day  |
        | 2013-05-09 12:00:00 |    1    |  day  |
        +---------------------+---------+-------+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: time

        >>> by_word = ts.group('words')
        >>> by_word.get_group('night')
        +---------------------+---------+-------+
        |         time        | numbers | words |
        +---------------------+---------+-------+
        | 2013-05-07 00:00:00 |    0    | night |
        | 2013-05-08 00:00:00 |    0    | night |
        | 2013-05-09 00:00:00 |    0    | night |
        +---------------------+---------+-------+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: time

        >>> by_num = ts.group('numbers')
        >>> by_num.get_group(1)
        +---------------------+---------+-------+
        |         time        | numbers | words |
        +---------------------+---------+-------+
        | 2013-05-07 12:00:00 |    1    |  day  |
        | 2013-05-08 12:00:00 |    1    |  day  |
        | 2013-05-09 12:00:00 |    1    |  day  |
        +---------------------+---------+-------+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: time

        >>> by_both = ts.group(['numbers','words'])
        >>> by_both.get_group([1, 'day'])
        +---------------------+---------+-------+
        |         time        | numbers | words |
        +---------------------+---------+-------+
        | 2013-05-07 12:00:00 |    1    |  day  |
        | 2013-05-08 12:00:00 |    1    |  day  |
        | 2013-05-09 12:00:00 |    1    |  day  |
        +---------------------+---------+-------+
        [3 rows x 3 columns]
        The index column of the TimeSeries is: time

        >>> by_day = ts.group([ts.date_part.YEAR,
        ...                    ts.date_part.MONTH,
        ...                    ts.date_part.DAY])
        >>> by_day.get_group([2013,5,9])
        +---------------------+---------+-------+
        |         time        | numbers | words |
        +---------------------+---------+-------+
        | 2013-05-09 00:00:00 |    0    | night |
        | 2013-05-09 12:00:00 |    1    |  day  |
        +---------------------+---------+-------+
        [2 rows x 3 columns]
        The index column of the TimeSeries is: time
        """
        if not isinstance(name, list):
            name = [name]

        # HUGE hack to prevent list of ints from converting to list of floats
        # on C++ side
        name.append(None)
        src_sf = self._grouped_ts.get_group(name)
        try:
            src_sf.remove_columns(self._temp_col_names)
        except KeyError:
            pass
        return _graphlab.TimeSeries(src_sf, self.index_col_name,is_sorted=True)

    def groups(self):
        """
        The name of each group in the GroupedTimeSeries.

        Returns
        -------
        out : SArray
        """
        return self._grouped_ts.groups()

    def group_info(self):
        """
        Returns an SFrame that contains the key columns for the groups and
        tbe number of rows in each group.

        Returns
        -------
        out : SArray
        """
        return self._grouped_ts.group_info()

    def num_groups(self):
        """
        Number of groups in the GroupedTimeSeries.

        Returns
        -------
        out : int
            The number of groups
        """
        return self._grouped_ts.num_groups()

    def __iter__(self):
        """
        Iterator for the GroupedTimeSeries.

        Returns
        -------
        out : tuple
            A tuple with two elements: the group name and the TimeSeries that
            corresponds to that group name.
        """
        def generator():
            elems_at_a_time = 16
            self._grouped_ts.begin_iterator()
            ret = self._grouped_ts.iterator_get_next(elems_at_a_time)
            while(True):
                for j in ret:
                    try:
                        j[1].remove_columns(self._temp_col_names)
                    except KeyError:
                        pass
                    j[1] = _graphlab.TimeSeries(j[1], self.index_col_name,
                            is_sorted=True)
                    yield tuple(j)

                if len(ret) == elems_at_a_time:
                    ret = self._grouped_ts.iterator_get_next(elems_at_a_time)
                else:
                    break
        return generator()

    def __repr__(self):
        """
        Returns a string description of the GroupedTimeSeries.
        """
        groups = self.groups()
        num_groups = self.num_groups()
        if num_groups <= 7:
            groups_str = ', '.join(map(str, groups))
        else:
            groups_str = ', '.join(map(str, groups[0:3]))
            groups_str += '... (and %s more)' % num_groups

        ret = "GroupedTimeSeries grouped by %s\n" % self.key_columns
        ret += "--------------------------------------------------------------\n"
        ret += "Groups                 : %s\n" % groups_str
        ret += "Number of groups       : %s\n" % num_groups
        ret += "Number of rows (total) : %s\n" % \
                       self._grouped_ts.sframe.num_rows()
        ret += "Number of columns      : %s\n" % \
                       self._grouped_ts.sframe.num_columns()
        ret += '\n'
        return ret

    def __len__(self):
        '''
        Returns the number of groups in the GroupedTimeSeries.
        '''
        return self._sframe.__len__()

    def to_sframe(self, include_index = True):
        '''
        Convert the GroupedTimeSeries to an SFrame (zero copy).

        Note that the returned SFrame is sorted by the key columns followed by
        the index column.

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

          >>> data = grouped_ts.to_sframe()
          +---------------------+-------+
          |        index        | value |
          +---------------------+-------+
          | 2015-01-06 00:00:00 |   0   |
          | 2015-01-07 00:00:00 |   0   |
          | 2015-01-08 00:00:00 |   0   |
          | 2015-01-09 00:00:00 |   1   |
          | 2015-01-10 00:00:00 |   1   |
          | 2015-01-01 00:00:00 |   1   |
          | 2015-01-02 00:00:00 |   2   |
          | 2015-01-03 00:00:00 |   2   |
          | 2015-01-04 00:00:00 |   2   |
          | 2015-01-05 00:00:00 |   2   |
          +---------------------+-------+
          [366 rows x 2 columns]
        '''
        _mt._get_metric_tracker().track('grouped_timeseries.to_sframe')
        if include_index:
          return self._grouped_ts.sframe
        else:
          return self._grouped_ts.sframe.remove_column(self.index_col_name)

    def to_timeseries(self):
        '''
        Merge the collection of groups in the GroupedTimeSeries into a single
        Timeseries.

        Returns
        -------
        out : Timeseries:w


        Examples
        --------
        .. sourcecode:: python

          >>> data = grouped_ts.to_timeseries()
          +---------------------+-------+
          |        index        | value |
          +---------------------+-------+
          | 2015-01-01 00:00:00 |   0   |
          | 2015-01-02 00:00:00 |   1   |
          | 2015-01-03 00:00:00 |   0   |
          | 2015-01-04 00:00:00 |   1   |
          | 2015-01-05 00:00:00 |   0   |
          | 2015-01-06 00:00:00 |   1   |
          | 2015-01-07 00:00:00 |   0   |
          | 2015-01-08 00:00:00 |   1   |
          | 2015-01-09 00:00:00 |   0   |
          | 2015-01-10 00:00:00 |   1   |
          +---------------------+-------+
          [366 rows x 2 columns]
        '''
        _mt._get_metric_tracker().track('grouped_timeseries.to_timeseries')
        return _graphlab.TimeSeries(self._grouped_ts.sframe,
                                    index = self.index_col_name)
