"""This module is used to download and parse the log files from the Predictive
Service."""

def parse_logs(log_paths):
    """Loads multiple log files and parses them into a list of pairs.

    Parameters
    ----------

    log_paths : list of str
        A list of log_paths, which could include hdfs, s3, or local file
        paths.

    Returns
    -------

    list of (datetime, log) pairs, ordered by datetime.

    Example
    -------

    >>> import psclient, psclient.logparse
    >>> ps = psclient.connect()
    >>> psclient.logparse.parse_logs(ps.get_log_filenames('audit'))

    """
    from psclient.file_util import read as _read

    import datetime, json
    result = []
    for log_path in log_paths:
        data = _read(log_path).split('\n')
        for row in data:
            if not row: continue
            timestamp, log = row.split(',', 1)
            result.append((
                datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S"),
                json.loads(log)))

    # sort by timestamp
    result.sort()

    return result


def parse_logs_sframe(log_paths):
    """
    Fetch log files of type specified by log_affix and corresponding to time
    window specified by [start_time, end_time] and return in an SFrame.

    Parameters
    ----------

    log_paths : list of str

        A list of log_paths, which could include hdfs, s3, or local file
        paths.

    Returns
    -------

    SFrame

        The SFrame has rows of datetime and log.

    Example
    -------

    >>> import psclient, psclient.logparse
    >>> ps = psclient.connect()
    >>> psclient.logparse.parse_logs_sframe(ps.get_log_filenames('audit'))
    """
    import graphlab as _gl

    # 1. get a list of all files in log path matching log_type pattern
    # 2. filter those based on specified date range

    if len(log_paths) > 0:
        # for readability, partially apply read function
        read_log_file = functools.partial(_gl.SFrame.read_csv,
                                          header=False,
                                          column_type_hints=[str, dict])

        # create dummy first row to ensure types are correct for initializer
        init = _gl.SFrame({"X1": [""], "X2": [{}]})

        # fold each item onto the initial empty SFrame via SFrame.append
        start_time = time()
        gls_log_sf = reduce(lambda sf, f: sf.append(read_log_file(f)),
                            log_files, init)

        # remove dummy init row required for reduce
        gls_log_sf = gls_log_sf[1:]

        # convert datetime column to datetime type
        gls_log_sf["datetime"] = gls_log_sf["X1"]\
            .str_to_datetime("%Y-%m-%dT%H:%M:%S.%fZ")

        gls_log_sf.swap_columns("X1", "datetime")
        gls_log_sf.remove_column("X1")
        gls_log_sf.rename({"X2": "log"})

        # sort by timestamp
        gls_log_sf = gls_log_sf.sort("datetime")

        time_elapsed = time() - start_time
        _logger.info("Read, parsed, and merged %d log files in %d seconds" \
                     % (len(log_files), time_elapsed))

        return gls_log_sf

    return _gl.SFrame()
