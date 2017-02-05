"""
Top level `create` method for the anomaly detection toolkit. Automatically
determines the type of model and anomaly based on the data type.
"""

import datetime as _dt
import graphlab as _gl
import graphlab.connect as _mt
from graphlab.toolkits._main import ToolkitError as _ToolkitError


def create(dataset, features=None, verbose=True):
    """
    Create an anomaly detection model. Based on the type of the input data,
    this function automatically choose the anomaly detection model and the type
    of anomalies to search for. Generally speaking, if the input data appears
    to be a time series---if the dataset type is TimeSeries, one of the
    features is of type datetime.datetime, or there is only a single
    feature---the toolkit chooses the moving Z-score model.

    Parameters
    ----------
    dataset : SFrame or TimeSeries
        Input dataset. Determines the type of anomaly detection model and types
        of anomalies to search for.

    features : list[str], optional
        Names of columns in the input 'dataset' to use as features.

    verbose : bool, optional
        If True, print progress updates and model details.

    Returns
    -------
    model : GraphLab Create model

    See Also
    --------
    local_outlier_factor.create, graphlab.toolkits.dbscan.create

    Examples
    --------
    >>> sf = graphlab.SFrame({'x0': [0., 1., 1., 0., 1., 0., 5.],
    ...                       'x1': [2., 1., 0., 1., 2., 1.5, 2.5]})
    ...
    >>> m = graphlab.anomaly_detection.create(sf)
    >>> type(m)
    graphlab.toolkits.anomaly_detection.local_outlier_factor.LocalOutlierFactorModel
    ...
    >>> m['scores']
    +--------+----------------------+
    | row_id | local_outlier_factor |
    +--------+----------------------+
    |   2    |    0.951567102896    |
    |   0    |    0.951567102896    |
    |   5    |    1.00783754045     |
    |   4    |    0.982224576307    |
    |   3    |    1.05829898642     |
    |   1    |    1.05829898642     |
    |   6    |    2.52792223974     |
    +--------+----------------------+
    [7 rows x 2 columns]
    """
    _mt._get_metric_tracker().track('toolkit.anomaly_detection.create')

    ## Basic validation of the input dataset.
    if not isinstance(dataset, (_gl.SFrame, _gl.TimeSeries)):
        raise TypeError("Input 'dataset' must be an SFrame or TimeSeries.")

    if len(dataset) < 1 or len(dataset.column_names()) < 1:
        raise TypeError("Input 'dataset' is empty.")


    ## Figure out the features and do basic validation.
    if features is None:
        features = dataset.column_names()

    if (not isinstance(features, list) or
        not all([type(c) == str for c in features])):

        raise TypeError("If specified, input 'features' must be a list " +
                        "of strings.")

    if not all([c in dataset.column_names() for c in features]):
        raise _ToolkitError("The specified features could not all be found " +
                            "in the input 'dataset'.")


    ## If any valid features are datetime types LOF is not valid.
    ## If there is more than one feature Z-score is not valid.

    # Figure out if there is a datetime column.
    col_types = {k: v for k, v in zip(dataset.column_names(),
                                      dataset.column_types())}

    datetime_features = [c for c in features if col_types[c] == _dt.datetime]
    value_features = [c for c in features if col_types[c] != _dt.datetime]


    ## Decide which model to use.
    try_zscore = False

    if isinstance(dataset, _gl.TimeSeries):
        try_zscore = True

    else:  # dataset is an SFrame
        if len(datetime_features) > 0:
            try_zscore = True

        if len(value_features) == 1 and (col_types[value_features[0]] in (int, float)):
            try_zscore = True


    ## Create the relevant model.
    bandwidth = max(1, int(0.05 * len(dataset)))

    if try_zscore:
        if len(value_features) != 1 or len(datetime_features) > 1:
            raise _ToolkitError("Cannot select an appropriate anomaly " +
                                "detection model. For a " +
                                "local outlier factor model, please remove " +
                                "any datetime-type features. For a moving" +
                                "Z-score model, please identify one data" +
                                "feature (integer- or float-type) and at most" +
                                "one datetime column as an index (this indexing is done" +
                                "automatically for TimeSeries objects)")

        if isinstance(dataset, _gl.SFrame) and len(datetime_features) == 1:
            _dataset = _gl.TimeSeries(dataset, index=datetime_features[0])
        else:
            _dataset = dataset[:]

        if verbose:
            print("Creating a moving Z-score anomaly detection model.")

        model = _gl.moving_zscore.create(dataset=_dataset,
                                         feature=value_features[0],
                                         window_size=bandwidth,
                                         verbose=verbose)


    ## If not doing the moving z-score, do local outlier factor.
    else:
        if verbose:
            print("Creating a local outlier factor model.")

        model = _gl.local_outlier_factor.create(dataset=dataset,
                                                features=features,
                                                num_neighbors=bandwidth,
                                                verbose=verbose)

    return model
