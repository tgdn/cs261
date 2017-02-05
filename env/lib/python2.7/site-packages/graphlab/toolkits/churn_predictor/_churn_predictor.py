"""
This module contains the churn prediction high-level toolkit
"""

import graphlab as _gl
import graphlab.connect as _mt
import types as _types
import datetime as _datetime
import time as _time
import logging as _logging
from collections import OrderedDict as _OrderedDict

from graphlab.util import _raise_error_if_not_of_type
from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits._model import SDKModel as _SDKModel
from graphlab.toolkits._internal_utils import \
      _toolkit_repr_print, _precomputed_field, \
      _raise_error_if_not_sframe, _check_categorical_option_type

from graphlab._beta.views import OverviewApp as _OverviewApp
from graphlab._beta.views import ChurnPredictorDescriptionView as _model_description
from graphlab._beta.views import ChurnPredictorEvaluateView as _ChurnPredictorEvaluateView
from graphlab._beta.views import ChurnPredictorExploreView as _ChurnPredictorExploreView

_NoneType = type(None)
_LINE_MSG = "--------------------------------------------------"

#-----------------------------------------------------------------------------
#
#                             Utilities
#
#-----------------------------------------------------------------------------
def _time_le(t1, t2):
    if (t1.tzinfo != None and t2.tzinfo == None) or(t1.tzinfo == None and
            t2.tzinfo != None):
        raise _ToolkitError(
           "Parameter `time_boundaries` have times with different time zone "
           " than the provided data.")
    else:
        return t1 <= t2

def _get_model_from_proxy(proxy):
    from graphlab.toolkits.classifier.boosted_trees_classifier\
            import BoostedTreesClassifier as _BoostedTreesClassifier
    return _BoostedTreesClassifier(proxy)
def _get_explanation_model_from_proxy(proxy):
    from graphlab.toolkits.regression.decision_tree_regression\
            import DecisionTreeRegression as _DecisionTreeRegression
    return _DecisionTreeRegression(proxy)

def _print_with_progress(msg, verbose):
    if (verbose):
        print("PROGRESS: %s" % msg)

def _unix_timestamp_to_datetime(x):
    return _datetime.datetime.fromtimestamp(x)

def _time_to_unix_timestamp(dt):
    return int((dt - _datetime.datetime(1970, 1, 1, tzinfo = dt.tzinfo)).total_seconds())

def _timedelta_to_seconds(td):
    return int(td.total_seconds())

def _raise_error_if_not_timeseries(data, name = "dataset"):
    if not isinstance(data, _gl.TimeSeries):
        msg = "Input {name} must be a TimeSeries. "
        if isinstance(data, _gl.SFrame):
            msg += "An SFrame can be converted to a time-series as follows:\n"
            msg += "\ttimeseries = gl.TimeSeries({name}, index = 'INDEX_COL')"
        msg = msg.format(name = name)
        raise TypeError(msg)

def _get_time_boundaries(min_time, max_time, time_period, verbose):
    stpcnt = 10
    if max_time == min_time:
        raise _ToolkitError(
            "All the timestamps in the time-series are the same.")

    step = (max_time - min_time) / stpcnt
    _print_with_progress("No time boundaries specified, computing 10"
            " boundaries from %s to %s" % (min_time, max_time), verbose)
    if step <= time_period:
        raise _ToolkitError("Not enough time in the training data. "
                      "There should be more than %s time periods." % stpcnt)
    time_boundaries = [(i + 1) * step + min_time for i in range(stpcnt - 1)]
    return map(_time_to_unix_timestamp, time_boundaries)


# Decorator to annotate that this function is a "public" function
# (for a view) but hidden to a user.
# For all practical purposes, treat this as a backwards compatible function.
def _viewPublic(f):
    return f

#-----------------------------------------------------------------------------
#
#                       Create function
#
#-----------------------------------------------------------------------------
def create(observation_data,
           user_id='user_id',
           features=None,
           user_data=None,
           churn_period=_datetime.timedelta(days=30),
           grace_period=_datetime.timedelta(days=0),
           time_period=_datetime.timedelta(days=1),
           lookback_periods=[7, 14, 21, 60, 90],
           time_boundaries=None,
           is_data_aggregated=False,
           use_advanced_features = False,
           verbose=True,
           **kwargs):
    """
    Create a model to predict the probability that an active user will become
    inactive in the future.

    Parameters
    ----------

    observation_data : TimeSeries
        Activity data used to train a churn prediction model. The dataset can
        be of two forms:

          - **Raw log data** : Each row contains action(s) by a single user at
            a given time.
          - **Aggregated log data**: Each row contains aggregated statistics
            based on all action(s) performed over a time period.

        The input data must contain a column with the `user_id` that which is
        unique for each user.  User activity columns of type 'str' will be
        considered categorical.  Columns of type 'int' and 'float' will be
        considered numerical.

    user_id : string, optional
        The name of the column in `observation_data` that corresponds to the
        `user_id` (a unique ID for each user). The `user_id` column must be of
        type 'int' or 'str'.

    features : List of string, optional
        List of columns in `observation_data` to use as features while training
        the model. Set to `None` to use all columns. The feature columns
        provided here are used as inputs to a feature engineering process which
        generates valuable features used in the model training process.

        The feature columns can be of the following type (Columns of other
        types are not supported).

          - *Numeric*: values of numeric type (integer or float).
          - *Categorical*: values of type (string).

        The features provided here are transformed by performing various
        aggregations (over various periods of time). These aggregates are then
        used as inputs to the model training process. After the model has
        completed training, you can access the transformed features using
        `model.processed_training_data`.

    user_data : SFrame, optional
        An SFrame with additional metadata for the users. This SFrame must also
        contain a column with the same `user_id` column. After all the rich
        feature engineering transformations are performed, the metadata is
        incorporated into the model by performing an `inner` join with the
        training data.

    churn_period : datetime.timedelta, optional
        The time duration of in-activity used to define whether or not a user
        has churned.

    grace_period : datetime.timedelta, optional
        *Advanced Modifier* This parameter allows specifying an amount of time
        after the time boundary where activity can happen and still be
        considered churned. This is useful for recurring services, for
        instance a monthly delivery service. It is possible that the last
        delivery event happens, but the subscription was terminated beforehand.
        In this case, setting a grace period of one month will account
        for this. For instance:

        - If the churn_period is set to 3 months, and the grace_period is set to
          0 (zero), a user will be considered churn if there was no
          activity in the 3 months period after the time_boundary.

        - If the churn_period is set to 3 months, and the grace_period is set to
          1 month, a user will be considered churn if there was or wasn't
          activity in the month following the time_boundary, and no activity
          in the following two months.

    time_period: datetime.timedelta, optional
        The time-scale/granularity at which features are computed. (Does not
        apply when `is_data_aggregated=True`).

        Re-sampling data is important because it defines the time-scale at
        which feature patterns can be learned.  For example, a `time_period` of
        1 day implies that all patterns or features used to train this model
        are computed at a time-scale which is a multiple of 1 day. Furthermore,
        the model training is much faster while working with data aggregated at
        the appropriate time period. Reduce this parameter (to 4 hours, or 6
        hours) if shorter time periods are of more importance in your domain.
        Stretch it out (to 3 days or a week) if longer time periods are more
        important in your domain.

    lookback_periods : list[int], optional
        The various multiples of `time_period` used while computing features.

        Each period in this list corresponds to a window of time (in the past)
        to look back into. For example, if the list contains [7, 14] (and
        `time_period` is 1 day), then features are computed based on weekly and
        biweekly patterns.

    time_boundaries : list[datetime.datetime], optional
        A list of various points in time (in the past) to train the churn
        prediction model on.

        At each time boundary, training features are computed using data
        present before the boundary. Data after the time boundary is used to
        compute training labels (churn or no churn). Multiple time boundaries
        can be used to create more training data. When set to None, we
        automatically select appropriate boundaries based on your
        `observation_data`.

    is_data_aggregated : boolean, optional
        When set to True, the `observation_data` is assumed to be already
        aggregated to a granularity level defined by `time_period`.

        **Note**: If set to True, only feature columns from `observation_data`
        of type int, and float are considered.

    use_advanced_features: boolean, optional
        When set to True, more advanced features are added. These features
        may be harder to explain/interpret but may result in a model with
        better predictive power.

    **kwargs: optional
        Additional options passed to the boosted tree model during training.
        Some of the useful options are:

           - max_depth: Defaults to 6.
           - max_iterations: Defaults to 10.
           - min_loss_reduction: Defaults to 0.0.

         See the API documentation for
         :class:`~graphlab.boosted_trees_classifier.BoostedTreesClassifier` for
         more details on the options that can be tweaked.

    verbose: boolean, optional
        When set to true, more status output is generated

    Returns
    -------
    out : ChurnPredictor
          A trained model of type
          :class:`~graphlab.churn_predictor.ChurnPredictor`.

    See Also
    --------
    ChurnPredictor

    Notes
    -----
    This churn forecast is made based on a classifier model trained on
    historical user activity logs. There are two stages during the model
    training phase;

        - **Phase 1**: Feature engineering. In this phase, features are generated
          using the provided activity data.
        - **Phase 2**: Machine learning model training. In this phase, the features
          computed are used to train a classifier model (using boosted trees).

    For Phase 1, this toolkit performs a series of extremely rich set of
    features based on:

        - aggregate statistics (over various periods of time) of features in the
          activity data.
        - simple patterns like change in the aggregate statistics over various
          periods of time.
        - user metadata (using the `user_data` parameter),

    After Phase 1, a classifier model is trained using gradient boosted trees.


    Examples
    --------

    .. sourcecode:: python

        >>> import graphlab as gl
        >>> import datetime

        # Load a data set.
        >>> sf = gl.SFrame(
        ... 'https://static.turi.com/datasets/churn-prediction/online_retail.csv')

        # Convert InvoiceDate from string to datetime.
        >>> import dateutil
        >>> from dateutil import parser
        >>> sf['InvoiceDate'] = sf['InvoiceDate'].apply(parser.parse)

        # Convert SFrame into TimeSeries.
        >>> time_series = gl.TimeSeries(sf, 'InvoiceDate')

        # Create a train-test split.
        >>> train, valid = gl.churn_predictor.random_split(time_series,
        ...           user_id='CustomerID', fraction=0.9)

        # Train a churn prediction model.
        >>> model = gl.churn_predictor.create(train, user_id='CustomerID',
        ...                       features = ['Quantity'])
    """
    _mt._get_metric_tracker().track('toolkits.churn_predictor.create')

    # Type checking of parameters.
    # ---------------------------------------------------------------------
    _raise_error_if_not_timeseries(observation_data, "observation_data")
    _raise_error_if_not_of_type(user_data, [_NoneType, _gl.SFrame],
            "user_data")
    _raise_error_if_not_of_type(user_id, [str], "user_id")
    _raise_error_if_not_of_type(features, [list, _NoneType], "features")
    _raise_error_if_not_of_type(time_period, [_datetime.timedelta],
            "time_period")
    _raise_error_if_not_of_type(churn_period, [_datetime.timedelta],
            "churn_period")
    _raise_error_if_not_of_type(lookback_periods, [list],
            "lookback_periods")
    _raise_error_if_not_of_type(time_boundaries, [_NoneType, list],
            "time_boundaries")
    _raise_error_if_not_of_type(is_data_aggregated, [bool],
            "is_data_aggregated")

    time_period_int = _timedelta_to_seconds(time_period)
    if (time_period_int <= 0):
        raise _ToolkitError("time_period must be positive.")
    if (len(observation_data) < 100):
        raise _ToolkitError(
                "Churn prediction requires at least 100 rows of activity.")
    churn_period_int = _timedelta_to_seconds(churn_period)
    if (churn_period_int <= 0):
        raise _ToolkitError("'churn_period' must be positive.")
    grace_period_int = _timedelta_to_seconds(grace_period)
    if (grace_period_int < 0):
        raise _ToolkitError("'grace_period' must be positive or zero.")
    if (grace_period_int >= churn_period_int):
        raise _ToolkitError(
                "'grace_period' must be less than the 'churn_period'.")

    if features is None:
        features = observation_data.column_names()
        features.remove(observation_data.index_col_name)
        if user_id in features:
            features.remove(user_id)

    for t in lookback_periods:
        if type(t) != int:
            raise TypeError(
                "Lookback periods must be of type list[int]")

    # Initialize the Churn prediction model with data grouped by user.
    # ---------------------------------------------------------------------
    min_time = observation_data.min_time
    max_time = observation_data.max_time
    proxy = _gl.extensions._ChurnPredictor()
    opts = {"index_column": observation_data.index_col_name,
             "user_id": user_id,
             "time_unit": 1,
             "features": features,
             "lookback_periods": lookback_periods,
             "time_period": time_period_int,
             "churn_period": churn_period_int,
             "grace_period": grace_period_int,
             "model_options": kwargs,
             "use_advanced_features": use_advanced_features,
             "max_time": _time_to_unix_timestamp(max_time),
             "is_data_aggregated": is_data_aggregated}
    proxy.init_model(observation_data.to_sframe(), opts)

    _print_with_progress("Grouping observation_data by user.", verbose)
    grp_sorted_data = _gl.GroupedTimeSeries(observation_data, user_id)
    sorted_data = grp_sorted_data.to_sframe()

    proxy._add_or_update_state({
        "num_users": grp_sorted_data.num_groups(),
        "num_observations": len(sorted_data),
        "num_features": len(features),
    });

    # First level aggregate: Aggregate by day.
    # ---------------------------------------------------------------------
    if not is_data_aggregated:
      _print_with_progress(
          "Resampling grouped observation_data by time-period %s." \
                                          % str(time_period), verbose)
    aggregated_by_time = proxy.aggregate_by_time(sorted_data, True)

    unix_timestamps = []
    if time_boundaries is not None:
        for t in time_boundaries:
            if type(t) != _datetime.datetime:
                raise TypeError(
                    "Time boundaries must be of type list[datetime.datetime]")
            if _time_le(max_time, t):
                raise _ToolkitError(
              "Time boundary %s exceeds the maximum time in the dataset" % t)

        unix_timestamps = map(_time_to_unix_timestamp, time_boundaries)
    else:
        unix_timestamps = _get_time_boundaries(min_time, max_time, time_period,
                verbose)
    unix_timestamps = list(unix_timestamps)
    if len(unix_timestamps) == 0:
        raise _ToolkitError(
            "At least one time boundary is needed for Churn prediction.")
    proxy.set_time_boundaries(unix_timestamps)

    # Second level aggregate: Make features based on daily aggregates.
    # ---------------------------------------------------------------------
    big_user_aggregate = None
    _print_with_progress("Generating features at time-boundaries.", verbose)
    _print_with_progress(_LINE_MSG, verbose)
    for time_boundary in unix_timestamps:
        _print_with_progress("Features for %s" \
                        % _unix_timestamp_to_datetime(time_boundary), verbose)
        user_aggregate = proxy.per_user_aggregate(aggregated_by_time,
                time_boundary)

        if big_user_aggregate is None:
            big_user_aggregate = user_aggregate
        else:
            big_user_aggregate = big_user_aggregate.append(user_aggregate)

    # Join user-data with the aggregated features.
    # ---------------------------------------------------------------------
    if user_data is not None:
        _print_with_progress(
                "Joining user_data with aggregated features.", verbose)
        _print_with_progress(_LINE_MSG, verbose)
        big_user_aggregate = big_user_aggregate.join(user_data, on=user_id,
                how="inner")
        if len(big_user_aggregate) == 0:
            raise _ToolkitError(
              "No data present after joining user_data with observation_data")

    _print_with_progress(
        "Training a classifier model.", verbose)

    # Train a model.
    # ---------------------------------------------------------------------
    proxy.train_model(big_user_aggregate)

    msg = []
    msg.append(_LINE_MSG)
    msg.append("Model training complete: Next steps")
    msg.append(_LINE_MSG)
    msg.append("(1) Evaluate the model at various timestamps in the past:")
    msg.append("      metrics = model.evaluate(data, time_in_past)")
    msg.append("(2) Make a churn forecast for a timestamp in the future:")
    msg.append("      predictions = model.predict(data, time_in_future)")
    for m in msg:
        _print_with_progress(m, verbose)

    return ChurnPredictor(proxy)

def random_split(observation_data,
                 user_id = 'user_id',
                 fraction = 0.90,
                 seed = None):
    """
    Randomly split an SFrame/TimeSeries into two SFrames/TimeSeries based on
    the `user_id` such that one split contains data for a `fraction` of the
    users while the second split contains all data for the rest of the users.

    Users are divided into two groups. The first group contains a `fraction` of
    the total users in `observation_data`. The second SFrame/TimeSeries
    contains the remaining users of the original SFrame.  Once the user splits
    are created, the original `observation_data` is split so that all data
    corresponding to a certain split of users is placed in the same split of
    `observation_data`.

    Parameters
    ----------
    observation_data : SFrame | TimeSeries
        Dataset to split. It must contain a column of user ids

    user_id : string
        The name of the column in `observation_data` that corresponds to the
        a unique identified for the user.

    fraction : float
        Fraction of the users to fetch for the first returned SFrame.  Must be
        between 0 and 1.

    seed : int, optional
        Seed for the random number generator used to split.

    Examples
    --------

    .. sourcecode:: python

        # Split the data so that train has 90% of the users.
        >>> train, valid = gl.churn_predictor.random_split(time_series,
        ...                           user_id='CustomerID',
        ...                           fraction=0.9)

    """
    _mt._get_metric_tracker().track('toolkits.churn_predictor.random_split')

    _raise_error_if_not_of_type(observation_data, [_gl.SFrame, _gl.TimeSeries],
            "observation_data")
    _raise_error_if_not_of_type(user_id, str, "user_id")
    _raise_error_if_not_of_type(fraction, float, "fraction")
    _raise_error_if_not_of_type(seed, [int, _NoneType], "seed")

    if user_id not in observation_data.column_names():
        raise _ToolkitError(
          "Input 'observation_data' must contain a column called %s." % user_id)

    unique_users = _gl.SFrame({'user': observation_data[user_id].unique()})
    chosen, not_chosen = unique_users.random_split(fraction, seed)
    train = observation_data.filter_by(chosen['user'], user_id)
    valid = observation_data.filter_by(not_chosen['user'], user_id)
    return (train, valid)

#-----------------------------------------------------------------------------
#
#                       Get reasons for churn.
#
#-----------------------------------------------------------------------------

def _make_period_string(period):
    if period == _datetime.timedelta(days = 1):
        return "day"
    if period == _datetime.timedelta(seconds = 3600):
        return "hour"

    total_seconds = period.total_seconds()
    if period.days > 1:
        num_days = int(total_seconds) / 86400
        if num_days * _datetime.timedelta(days = 1) == period:
            return "(%d day periods)" % num_days

    elif period.days == 0:
        num_hours = int(total_seconds) / 3600
        if num_hours * _datetime.timedelta(hours = 1) == period:
            return "(%d hour periods)" % num_hours
        if num_hours == 0:
            return "(%.1f second periods)" % total_seconds
        else:
            return "(%.1f hour periods)" % (total_seconds / 3600)

    # Default
    return "(%s periods)" % period

def _make_boundary_string(boundary):
    return str(boundary)

def _translate_path_to_english(path, period, boundary):
    names = _gl.extensions._churn_predictor_to_english_names_apply(
                                                    path, period, boundary)
    reasons = []
    for i in range(len(names)):
        sign = path[i]["sign"]
        val = path[i]["value"]
        name = names[i]
        if sign == "in":
            if (type(val[0]) in [int, float]) and (type(val[1]) in [int, float]):
              reason = name.format(
                  DECISION = "between %.2f and %.2f" % (val[0], val[1]))
            else:
              reason = name.format(
                  DECISION = "between %s and %s" % (val[0], val[1]))

        elif sign == ">=":
            if type(val) in [int, float]:
                reason = name.format(DECISION = "greater than (or equal to) %.2f" % val)
            else:
                reason = name.format(DECISION = "greater than (or equal to) %s" % val)
        elif sign == "<":
            if type(val) in [int, float]:
                reason = name.format(DECISION = "less than %.2f" % val)
            else:
                reason = name.format(DECISION = "less than %s" % val)

        elif sign == "=":
            if path[i]["node_type"] in ["float", "integer"]:
                if type(val) in [int, float]:
                    reason = name.format(DECISION = "is exactly %.2f" % val)
                else:
                    reason = name.format(DECISION = "is exactly %s" % val)
            else:
                reason = name.format(DECISION = "is")
        elif sign == "!=":
            reason = name.format(DECISION = "is not")
        elif sign == "missing":
            reason = name.format(DECISION = "")
        else:
            reason = name # Should not happen.

        # Sentence case it.
        reason = reason.strip()
        if len(reason) == 0:
            reason = ''
        else:
            reason  = ''.join([reason[0].upper() + reason[1:]])

        # Add to list of reasons
        reasons.append(reason)
    return list(set(reasons))

def _translate_path_to_annotated_english(path, period, boundary):
    for i in range(len(path)):
        sign = path[i]["sign"]
        val = path[i]["value"]
        if type(val) == tuple:
            path[i]["value"][0] = "\v%s\v " % val[0]
            path[i]["value"][1] = "\v%s\v " % val[1]
        if type(val) in [str, int, float]:
            path[i]["value"] = "\v%s\v" % val
    reasons = _translate_path_to_english(path, period, boundary)
    ret_reasons = []
    for reason in reasons:
        ret = []
        start = 0
        state = "TEXT"
        for i in range(len(reason)):
            if reason[i] == '"' and state == "TEXT":
                ret += [(reason[start:i], state)]
                start = i
                state = "FEATURE"
                continue
            if reason[i] == '"' and state == "FEATURE":
                ret += [(reason[start: i+1], state)]
                start = i + 1
                state = "TEXT"
                continue
            if reason[i] == "\v" and state == "TEXT":
                ret += [(reason[start:i], state)]
                start = i + 1
                state = "VALUE"
                continue
            if reason[i] == "\v" and state == "VALUE":
                ret += [(reason[start : i], state)]
                start = i + 1
                state = "TEXT"
                continue
        if start != len(reason) - 1:
            ret += [(reason[start:], state)]
        ret_reasons.append(ret)
    return ret_reasons

def _translate_feature_name_to_english(importance, period, boundary):
    def _make_sentence_case(x):
        if "{DECISION}" in x:
            x = x.format(DECISION = '').strip()
            return ''.join([x[0].upper() + x[1:]])
        else:
            return x
    importance_sa = importance.apply(lambda x:
                         [{'child_id': 1,
                          'feature': x['name'],
                          'index': x['index'],
                          'node_id': 0,
                          'node_type': 'description',
                          'sign': '',
                          'is_missing': 0,
                          'value': 0}])
    return _gl.extensions._churn_predictor_to_english_names(importance_sa,
            period, boundary).apply(lambda x: _make_sentence_case(x[0]))


#-----------------------------------------------------------------------------
#
#                             Churn predictor model.
#
#-----------------------------------------------------------------------------
_DEFAULT_OPTIONS = {
}

get_default_options = _get_default_options_wrapper(
    '_ChurnPredictor', 'churn_predictor', 'ChurnPredictor', True)
class ChurnPredictor(_SDKModel):
    """

    Create a churn forecast model (an object of type
    :class:`~graphlab.churn_predictor.ChurnPredictor`) that predicts the
    probability that an `active` user/customer will become `inactive` in the
    future (defined as churn). This churn forecast is made based on a
    classifier model trained on historical user activity logs.

    This model cannot be constructed directly.  Instead, use
    :func:`graphlab.churn_predictor.create` to create an instance of this
    model. More details about how to use this object are availiable in the
    documentation for the create function.

    See Also
    --------
    create

    Examples
    --------
    .. sourcecode:: python

        # Convert SFrame into TimeSeries.
        >>> time_series = gl.TimeSeries(sf, 'InvoiceDate')

        # Create a train-test split.
        >>> train, valid = gl.churn_predictor.random_split(time_series,
        ...                           user_id='CustomerID',
        ...                           fraction=0.9)


        # Period of inactivity that defines churn.
        >>> churn_period = datetime.timedelta(days = 30)

        # Train a churn prediction model.
        >>> model = gl.churn_predictor.create(train, user_id='CustomerID',
        ...                       features = ['Quantity'],
        ...                       churn_period = churn_period)

    """
    __proxy__ = None
    def __init__(self, model_proxy):
        self.__proxy__ = model_proxy

    def _get_wrapper(self):
        proxy_wrapper = self.__proxy__._get_wrapper()

        def model_wrapper(unity_proxy):
            model_proxy = proxy_wrapper(unity_proxy)
            return ChurnPredictor(model_proxy)
        return model_wrapper

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the model.
        """
        return self.__repr__()

    def _get_summary_struct(self):
        """
        Returns a structured description of the model, including (where relevant)
        the schema of the training data, description of the training data,
        training statistics, and model hyperparameters.

        Returns
        -------
        sections : list (of list of tuples)
            A list of summary sections.
              Each section is a list.
                Each item in a section list is a tuple of the form:
                  ('<label>','<field>')
        section_titles: list
            A list of section titles.
              The order matches that of the 'sections' object.
        """

        model_fields = [
            ('Number of observations', 'num_observations'),
            ('Number of users', 'num_users'),
            ('Number of feature columns', 'num_features'),
            ('Features used', 'features')
        ]

        _num_boundaries = _precomputed_field(len(self.get('time_boundaries')))
        _time_period = _precomputed_field(str(self.get('time_period')))
        _churn_period = _precomputed_field(str(self.get('churn_period')))
        hyperparam_fields = [
            ("Lookback periods", 'lookback_periods'),
            ("Number of time boundaries", _num_boundaries),
            ("Time period", _time_period),
            ("Churn period", _churn_period)
        ]
        return ([model_fields, hyperparam_fields], ['Schema', 'Parameters'])


    def __repr__(self):
        """
        Print a string description of the model, when the model name is entered
        in the terminal.
        """

        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    def get(self, field):
        """
        Return the value of a given field. The list of all queryable fields is
        detailed below, and can be obtained programmatically with the
        :func:`~graphlab.churn_predictor.ChurnPredictor.list_fields`
        method.

        +---------------------------+-------------------------------------------------------------+
        |      Field                | Description                                                 |
        +===========================+=============================================================+
        | churn_period              | Period of in-activity used to define churn.                 |
        +---------------------------+-------------------------------------------------------------+
        | grace_period              | Period within the churn_period where churn is allowed       |
        +---------------------------+-------------------------------------------------------------+
        | categorical_features      | List of features treated as categorical features.           |
        +---------------------------+-------------------------------------------------------------+
        | numerical_features        | List of features treated as numerical features.             |
        +---------------------------+-------------------------------------------------------------+
        | features                  | List of all features used in the model.                     |
        +---------------------------+-------------------------------------------------------------+
        | is_data_aggregated        | Is the input data already in an aggregated format?          |
        +---------------------------+-------------------------------------------------------------+
        | lookback_periods          | Periods to lookback into the past during feature creation.  |
        +---------------------------+-------------------------------------------------------------+
        | model_options             | Additional options provided to the boosted tree classifier. |
        +---------------------------+-------------------------------------------------------------+
        | num_features              | Number of features used while training the model.           |
        +---------------------------+-------------------------------------------------------------+
        | num_observations          | Number of observations in the training data.                |
        +---------------------------+-------------------------------------------------------------+
        | num_users                 | Number of users in the training data.                       |
        +---------------------------+-------------------------------------------------------------+
        | processed_training_data   | Output of the feature engineering steps.                    |
        +---------------------------+-------------------------------------------------------------+
        | time_boundaries           | Churn boundaries used while training the model.             |
        +---------------------------+-------------------------------------------------------------+
        | time_period               | Time-scale at which patterns are learned.                   |
        +---------------------------+-------------------------------------------------------------+
        | user_id                   | Column name with the user id.                               |
        +---------------------------+-------------------------------------------------------------+
        | trained_model             | Trained model used for predictions                          |
        +---------------------------+-------------------------------------------------------------+
        | trained_explanation_model | Trained model used for generating segments/explanations     |
        +---------------------------+-------------------------------------------------------------+

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out
            Value of the requested fields.

        See Also
        --------
        list_fields

        Examples
        --------
        .. sourcecode:: python

            >>> model = graphlab.logistic_classifier.create(data,
            ...                target='is_expensive', features=['bedroom', 'size'])

            # Data computed at the end of feature engineering.
            >>> feature_engineered_data = model.processed_training_data
        """
        _mt._get_metric_tracker().track('toolkits.churn_predictor.get')
        ret = self.__proxy__.get(field)
        if field == 'time_period':
            import datetime
            return datetime.timedelta(seconds = ret)
        elif field == 'churn_period':
            import datetime
            return datetime.timedelta(seconds = ret)
        elif field == 'grace_period':
            import datetime
            return datetime.timedelta(seconds = ret)
        elif field == 'trained_model':
            return _get_model_from_proxy(ret)
        elif field == 'trained_explanation_model':
            return _get_explanation_model_from_proxy(ret)
        elif field == 'model_options':
            if "_internal_opts" in ret:
                ret.pop("_internal_opts")
            return ret
        else:
            return ret

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and train the model.

        Returns
        -------
        out : dict
            Dictionary with options used to define and train the model.

        See Also
        --------
        get_default_options, list_fields, get

        Examples
        --------
        .. sourcecode:: python

            >>> print model.get_current_options()
            {'model_options': {'max_depth': 100, 'min_loss_reduction': 0.0}}
        """

        _mt._get_metric_tracker().track(\
                  'toolkits.churn_predictor.get_current_options')
        return self.__proxy__.get_current_options()

    def list_fields(self):
        """
        List the fields stored in the model, including data, model, and training
        options. Each field can be queried with the ``get`` method.

        Returns
        -------
        out : list
            List of fields queryable with the ``get`` method.

        See Also
        --------
        get

        Examples
        --------
        .. sourcecode:: python

            >>> print model.list_fields()
                ['categorical_features',
                 'churn_period',
                 'grace_period',
                 'features',
                 'is_data_aggregated',
                 'lookback_periods',
                 'model_options',
                 'num_features',
                 'num_observations',
                 'num_users',
                 'numerical_features',
                 'prediction_window',
                 'processed_training_data',
                 'time_boundaries',
                 'time_period',
                 'user_id',
                 'trained_model']
        """

        _mt._get_metric_tracker().track(
                'toolkits.churn_predictor.list_fields')
        return self.__proxy__.list_fields()

    def extract_features(self,
                dataset,
                time_boundary=None,
                user_data=None,
                verbose=True):
        """
        Transforms an input timeseries (and optional per-user data) and
        generates features that capture per-user behavior upto a given time
        boundary. By default, the time-boundary is set to the last date
        in the input timeseries.

        Parameters
        ----------
        dataset : TimeSeries
            A dataset that has the same columns that were used during training.
            See the create function for more details on the format of the
            dataset.G

        time_boundary : datetime, optional
            Time-boundary at which features are generated. By default the last
            timestamp of the `dataset` is used.

            The time boundary is used to determine features and a infer a label
            for churn. All the data present before the boundary will be used to
            compute features, and the data after the boundary will be used
            to infer whether or not a user will churn. Data after the boundary
            will not be used to generate features.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column with
            the same name as what is specified by the `user_id` input parameter.
            `user_data` can provide any amount of additional user-specific
            information. The join performed is an inner join.


        verbose: boolean, optional
            When set to True, more status messages are displayed.

        Returns
        -------
        out : (SFrame, SFrame) Tuple of SFrame.
            The first SFrame contains features computed based on interaction
            data of users up to the time_boundary.

            The second SFrame contains the sets of users for which there isn't
            enough interaction data to compute features.

        See Also
        --------
        create

        Examples
        --------

        You can perform the same set of feature transformations on a new dataset
        using the following function:

        .. sourcecode:: python

            # Choose a period in time one month before the final time.
            >>> one_month_before = time_series.max_time - datetime.timedelta(
            ...                        days = 30)

            # Extract features based on this point of time.
            # The SFrame features contains all the temporal features extracted
            # for the purpose of churn prediction.
            >>> features, bad_users = model.extract_features(time_series)


            # The second SFrame contains a list of users that don't have enough
            # activity at a period in time before the time boundary.
            >>> print bad_users
            +------------+
            | CustomerID |
            +------------+
            |   18209    |
            |   18139    |
            |   18084    |
            |   18082    |
            |   18058    |
            |   18034    |
            |   18014    |
            |   18006    |
            |   17985    |
            |   17936    |
            +------------+
            [230 rows x 1 columns]

            """
        _raise_error_if_not_timeseries(dataset, "dataset")
        _raise_error_if_not_of_type(time_boundary,
            [_NoneType, _datetime.datetime], "time_boundary")
        _raise_error_if_not_of_type(user_data, [_NoneType, _gl.SFrame],
                "user_data")

        # Initialize the Churn prediction model with data grouped by user.
        # ---------------------------------------------------------------------
        _print_with_progress("Grouping dataset by user.", verbose)
        user_id = self.get("user_id")
        sorted_data_grp = _gl.GroupedTimeSeries(dataset, user_id)
        unique_users = _gl.SFrame({user_id: sorted_data_grp.groups()})
        sorted_data = sorted_data_grp.to_sframe()

        # First level aggregate: Aggregate by day.
        # ---------------------------------------------------------------------
        proxy = self.__proxy__
        time_period_disp = self.get("time_period")
        time_period = int(time_period_disp.total_seconds())
        if not self.get("is_data_aggregated"):
            _print_with_progress(
                "Resampling grouped observation_data by time-period %s." \
                                          % str(time_period_disp), verbose)
        aggregated_by_time = proxy.aggregate_by_time(sorted_data, False)

        # Second level aggregate: Make features based on daily aggregates.
        # ---------------------------------------------------------------------
        if time_boundary is None:
            time_boundary = dataset.max_time
        else:
            if _time_le(time_boundary, dataset.min_time):
                raise _ToolkitError("time_boundary cannot be before the first" +
                                            " timestamp in the input dataset.")

        _print_with_progress("Generating features for boundary %s." %\
                                    time_boundary, verbose)
        prepare_time = _time_to_unix_timestamp(time_boundary) + time_period
        eval_data = proxy.per_user_aggregate(aggregated_by_time,
                prepare_time)
        bad_users = unique_users.filter_by(eval_data[user_id], user_id, True)

        # Join user-data with the aggregated features.
        # ---------------------------------------------------------------------
        if user_data is not None:
            _print_with_progress(
                    "Joining user_data with aggregated features.", verbose)
            eval_data = eval_data.join(user_data, on=user_id, how="inner")
        return eval_data, bad_users

    @_viewPublic
    def _internal_predict(self, dataset, time_boundary=None, user_data=None,
            include_leaf_id = False, include_label = False,
            include_bad_users = True, verbose=True):
        """
        Internal predict method that is re-used by the churn report as well
        as a predict functions.

        Parameters
        ----------
        dataset :TimeSeries
            A dataset that has the same columns that were used during training.

        time_boundary : datetime, optional
            Time-boundary strictly after which a prediction for churn/stay is
            made. By default the last timestamp of the `dataset` is used.

            The time boundary is used to compute features as well as make a
            prediction for whether or not the user will churn/stay after the
            time-boundary. All the data present before the boundary will be
            used to compute features, and the data after the boundary will be
            ignored.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column
            with the same name as what is specified by the `user_id` input
            parameter.  `user_data` can provide any amount of additional
            user-specific information. The join performed is an inner join.

        include_leaf_id: bool, optional
            Include leaf_id for predictions? (used by churn report)

        include_label: bool, optional
            Include label for evaluation?

        include_bad_users: bool, optional
            Include users without enough data?

        verbose: boolean, optional
            When set to true, more status output is generated
            Default: True

        """
        _raise_error_if_not_timeseries(dataset, "dataset")
        _raise_error_if_not_of_type(time_boundary,
            [_NoneType, _datetime.datetime], "time_boundary")
        _raise_error_if_not_of_type(user_data, [_NoneType, _gl.SFrame],
                "user_data")

        if time_boundary is None:
            start_time = dataset.max_time
        else:
            start_time = time_boundary
        end_time = start_time + self.get("churn_period")
        _print_with_progress("Making a churn forecast for the time window:",
                                            verbose)
        _print_with_progress(_LINE_MSG, verbose)
        _print_with_progress(" Start : %s" % start_time, verbose)
        _print_with_progress(" End   : %s" % end_time, verbose)
        _print_with_progress(_LINE_MSG, verbose)

        # Call extract features.
        # ---------------------------------------------------------------------
        proxy = self.__proxy__
        features_data, bad_users = self.extract_features(dataset,
                time_boundary, user_data, verbose)
        num_bad_users = len(bad_users)

        # Make prediction from the model.
        # ---------------------------------------------------------------------
        predictions = proxy.predict(features_data)
        if (len(bad_users) > 0):
            _print_with_progress(
                "Not enough data to make predictions for %s user(s). "\
                     % len(bad_users), verbose)
        bad_users['probability'] = _gl.SArray.from_const(None,
                num_bad_users).astype(float)

        # Add the leaf id from the first tree (for churn report).
        # -----------------------------------------------------------------
        if include_leaf_id:
            tree_model = self.trained_explanation_model
            predictions.add_columns(
                tree_model._extract_features_with_missing(features_data))

            bad_users['leaf_id'] = _gl.SArray.from_const(None,
                    num_bad_users).astype(int)
            bad_users['missing_id'] = _gl.SArray.from_const([],
                    num_bad_users).astype(list)


        # Include the label (used for evaluation)
        # -----------------------------------------------------------------
        if include_label:
            predictions['label'] = features_data['__internal_label']
            bad_users['label'] = _gl.SArray.from_const(None,
                    num_bad_users).astype(int)

        if include_bad_users:
            if len(bad_users) > 0:
                predictions = predictions.append(bad_users)
        return predictions

    def predict(self,
                dataset,
                time_boundary=None,
                user_data=None,
                verbose=True):
        """
        Use the trained model to make a forecast of churn (along with a
        probability of churn for each user) at a given period of time.

        **Making a forecast**.

        By default, the model will make a forecast of churn proabability for a
        `churn_period` of time strictly after the last timestamp in the
        `dataset`.

        **Making predictions for post-hoc analysis.**

        Sometimes, it is useful to simulate how the model would have performed
        at a period of time in the past. For this, you can set the
        `time_boundary` to a point in time in the past for which the model is
        expected to make predictions.

        Parameters
        ----------
        dataset :TimeSeries
            A dataset that has the same columns that were used during training.

        time_boundary : datetime, optional
            Time-boundary strictly after which a prediction for churn/stay is
            made. By default the last timestamp of the `dataset` is used.

            The time boundary is used to compute features as well as make a
            prediction for whether or not the user will churn/stay after the
            time-boundary. All the data present before the boundary will be
            used to compute features, and the data after the boundary will be
            ignored.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column
            with the same name as what is specified by the `user_id` input
            parameter.  `user_data` can provide any amount of additional
            user-specific information. The join performed is an inner join.

        verbose: boolean, optional
            When set to true, more status output is generated
            Default: True

        Returns
        -------
        out : SFrame
            An SFrame with model predictions. A prediction of 100% means the
            user is likely to stay, whereas a prediction of 0% means the
            user is likely to leave (churn).

        See Also
        --------
        create

        Examples
        --------

        .. sourcecode:: python

            # Make a prediction (into the future) for all users in the dataset.
            >>> model.predict(time_series)
            +------------+-----------------+
            | CustomerID |   probability   |
            +------------+-----------------+
            |   12346    |  0.67390537262  |
            |   12347    |  0.760785758495 |
            |   12348    |  0.62475168705  |
            |   12349    |  0.372504591942 |
            |   12350    |  0.67390537262  |
            |   12352    |  0.201043695211 |
            |   12353    |  0.821378648281 |
            |   12354    |  0.737500548363 |
            |   12355    |  0.699232280254 |
            +------------+-----------------+
            [4340 rows x 2 columns]

            # The predictions from the model are only valid to predict inactivity
            # during this time period.
            >>> prediction_window = [time_series.max_time,
            ...             time_series.max_time + model.churn_period]
            >>> print prediction_window
                   [datetime.datetime(2011, 10, 1, 7, 0),
                    datetime.datetime(2011, 10, 31, 7, 0)]

        """
        _mt._get_metric_tracker().track('toolkits.churn_predictor.predict')
        return self._internal_predict(dataset, time_boundary=time_boundary,
                user_data=user_data, include_leaf_id = False, verbose=verbose)

    def _get_explanation_tree(self):
        return self.trained_explanation_model._get_tree(0)

    def explain(self,
                dataset,
                time_boundary=None,
                user_data=None,
                verbose=True):
        """
        Use the model to make predictions and also provide explanations of
        why the prediction was made.

        Parameters
        ----------
        dataset :TimeSeries
            A dataset that has the same columns that were used during training.

        time_boundary : datetime, optional
            Time-boundary strictly after which a prediction for churn/stay is
            made. By default the last timestamp of the `dataset` is used.

            The time boundary is used to compute features as well as make a
            prediction for whether or not the user will churn/stay after the
            time-boundary. All the data present before the boundary will be
            used to compute features, and the data after the boundary will be
            ignored.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column
            with the same name as what is specified by the `user_id` input
            parameter.  `user_data` can provide any amount of additional
            user-specific information. The join performed is an inner join.

        verbose: boolean, optional
            When set to true, more status output is generated
            Default: True

        Returns
        -------
        out : SFrame
            An SFrame with a list of explainations for each prediction.

        See Also
        --------
        create

        Examples
        --------

        .. sourcecode:: python

            # Make a prediction (into the future) for all users in the dataset.
            >>> model.explain(time_series)
        """
        _mt._get_metric_tracker().track('toolkits.churn_predictor.predict')
        predictions = self._internal_predict(dataset, time_boundary=time_boundary,
                user_data=user_data, include_leaf_id = True, verbose=verbose)

        # Extract the tree.
        tree = self._get_explanation_tree()
        period = _make_period_string(self.time_period)
        boundary = dataset.max_time if time_boundary is None else time_boundary
        boundary = "start of forecast"

        def get_reason_for_churn(x):
            path = tree.get_prediction_path(x['leaf_id'], x['missing_id'])
            return _translate_path_to_english(path, period, boundary)

        predictions['explanation'] = predictions.apply(get_reason_for_churn)
        predictions = predictions.remove_columns(['leaf_id', 'missing_id'])
        return predictions


    def get_churn_report(self, dataset, time_boundary=None, user_data=None,
            max_segments = None, verbose=True):
        """
        Get a detailed churn report which clusters the users into segments
        and then provides a reason for churn for users in the segment.

        Parameters
        ----------
        dataset :TimeSeries
            A dataset that has the same columns that were used during training.

        time_boundary : datetime, optional
            Time-boundary strictly after which a prediction for churn/stay is
            made. By default the last timestamp of the `dataset` is used.

            The time boundary is used to compute features as well as make a
            prediction for whether or not the user will churn/stay after the
            time-boundary. All the data present before the boundary will be
            used to compute features, and the data after the boundary will be
            ignored.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column
            with the same name as what is specified by the `user_id` input
            parameter.  `user_data` can provide any amount of additional
            user-specific information. The join performed is an inner join.

        max_segments: int, optional
            Limit the number of customer churn segments in the output.  If set
            to None, then there isn't any limit imposed. If set to a finite
            integer, the most significant customer segments are returned along
            with 2 miscellaneous segments (one churning) and the other (not
            churning).

        verbose: boolean, optional
            When set to true, more status output is generated
            Default: True

        Returns
        -------
        out: SFrame

        An SFrame which contains the following columns
         - segment_id           : A segment id for a group of users.
         - num_users            : Number of users in this segment.
         - num_users_percentage : Number of users in this segment.
         - explanation          : An explanation for why the model made the
                                  prediction.
         - avg_probability      : Avg churn probability for all users in this
                                  segment.
         - stdv_probability     : Stdv of churn probability scores.
         - users                : List of all users in this segment.

        Examples
        --------

        .. sourcecode:: python

            # Make a prediction (into the future) for all users in the dataset.
            >>> segments = model.get_churn_report(time_series)

        """
        _mt._get_metric_tracker().track(
                       'toolkits.churn_predictor.get_churn_report')
        # Call predict.
        predictions = self._internal_predict(dataset, time_boundary=time_boundary,
                user_data=user_data, include_leaf_id = True, verbose=verbose)

        # Segment the data based on the predictions.
        segments = self._get_segments_from_predictions(predictions,
                max_segments)

        # Prepare a churn report from the segments.
        boundary = dataset.max_time if time_boundary is None else time_boundary
        segments = self._prepare_churn_report(segments, boundary)

        # The segment info is confusing for this use case. Remove it!
        # Also, make sure the column order is sensible.
        order = ['segment_id', 'num_users', 'num_users_percentage',
                'explanation', 'avg_probability', 'stdv_probability', 'users']
        return segments[order].sort('num_users', ascending = False)

    @_viewPublic
    def _get_segments_from_predictions(self, predictions, max_segments = 8):
        """
        Internal function to take the predictions (with leaf id) and then
        segment it.
        """
        if max_segments is None:
            max_segments = len(predictions)

        # Segment your data.
        predictions = predictions.dropna('probability')
        segments = predictions.groupby(['leaf_id', 'missing_id'], {
                      'num_users'   : _gl.aggregate.COUNT(),
                      'probability' : _gl.aggregate.AVG('probability'),
                   })
        segments = segments.sort('num_users', ascending = False)
        segments = segments.add_row_number()
        def segment_name(x):
            row = x['id']
            prob = x['probability']
            if row >= max_segments:
                if prob >= 0.5:
                    return 'Other (Churn)'
                else:
                    return 'Other (Active)'
            else:
                return '%s' % row
        segments['segment_id'] = segments.apply(segment_name)

        # Join it back and compute more advanced stats about the leaf.
        return predictions.join(
                   segments[['leaf_id', 'missing_id', 'segment_id']])

    @_viewPublic
    def _prepare_churn_report(self, segments, boundary, annotation=False):
        """
        Internal function to prepare a churn report from the segments.
        """
        # Aggregate segments
        segments = segments.pack_columns(['leaf_id', 'missing_id'],
                        new_column_name='leaf_missing_id')
        segments['leaf_missing_id'] = segments['leaf_missing_id'].astype(str)

        segments  = segments.groupby('segment_id', {
             'num_users'        : _gl.aggregate.COUNT(),
             'users'            : _gl.aggregate.CONCAT(self.user_id),
             'segment_info'     : _gl.aggregate.FREQ_COUNT('leaf_missing_id'),
             'avg_probability'  : _gl.aggregate.AVG('probability'),
             'min_probability'  : _gl.aggregate.MIN('probability'),
             'max_probability'  : _gl.aggregate.MAX('probability'),
             'stdv_probability' : _gl.aggregate.STDV('probability'),
        })
        total_users = segments['num_users'].sum()
        segments['num_users_percentage'] = \
                          100.0 * segments['num_users'] / total_users

        # Extract the tree.
        tree = self._get_explanation_tree()

        # Add the churn-reason.
        period = _make_period_string(self.time_period)
        boundary = "start of forecast"
        def get_churn_reason(x):
            if len(x) > 1:
                return None
            import json
            leaf_id, missing_id = json.loads(list(x.keys())[0])
            path = tree.get_prediction_path(leaf_id, missing_id)
            return _translate_path_to_english(path, period, boundary)

        def get_annotated_churn_reason(x):
            if len(x) > 1:
                return None
            import json
            leaf_id, missing_id = json.loads(list(x.keys())[0])
            path = tree.get_prediction_path(leaf_id, missing_id)
            return _translate_path_to_annotated_english(path, period, boundary)


        segments['explanation'] = segments['segment_info'].apply(get_churn_reason)
        if annotation == True:
            segments['annotated_explanation'] = \
                    segments['segment_info'].apply(get_annotated_churn_reason)
        return segments

    @_viewPublic
    def _evaluation_prepare(self,
                dataset,
                time_boundary,
                user_data=None,
                verbose=True):
        """
        Prepare the data to evaluate a trained model given the data present in
        ``dataset``. The evaluation dataset can be constructed by taking a
        sample of users from the same source as the training data.

        Parameters
        ----------
        dataset : SFrame
            A dataset that has the same columns that were used during training.

        time_boundary : datetime
            Time to use as a boundary for evaluation. User activity beyond that
            point will be used to determine whether users have churned.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column with
            the same name as what is specified by the `user_id` input parameter.
            `user_data` can provide any amount of additional user-specific
            information. The join performed is an inner join.

        verbose: boolean, optional
            When set to true, more status output is generated
            Default: True

        Returns
        -------
        out : SFrame

        See Also
        --------
        create
        """
        _raise_error_if_not_timeseries(dataset, "dataset")
        _raise_error_if_not_of_type(user_data, [_NoneType, _gl.SFrame],
                "user_data")
        _raise_error_if_not_of_type(time_boundary, [_datetime.datetime],
                "time_boundary")
        return self._internal_predict(dataset, time_boundary=time_boundary,
                user_data=user_data, include_leaf_id = False, include_label =
                True, include_bad_users = False, verbose=verbose)

    def evaluate(self,
                dataset,
                time_boundary,
                user_data=None,
                metric='auto',
                cutoffs=[0.1, 0.25, 0.5, 0.75, 0.9],
                verbose=True):
        """
        Evaluate a trained model given the data present in ``dataset``. The
        evaluation dataset can be constructed by taking a sample of users from
        the same source as the training data.

        Parameters
        ----------
        dataset : SFrame
            A dataset that has the same columns that were used during training.

        time_boundary: datetime
            Time to use as a boundary for evaluation. User activity beyond that
            point will be used to determine whether users have churned.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column with
            the same name as what is specified by the `user_id` input parameter.
            `user_data` can provide any amount of additional user-specific
            information. The join performed is an inner join.

        metric: str | list[str] optional
            Select the metrics. If None, will run all the metrics. Available
            metrics are:

                - 'roc_curve'                : The ROC curve.
                - 'precision_recall_curve'   : Precision-recall curve.
                - 'auc'                      : The area under the ROC curve.
                - 'precision'                : Precision at cutoff = 0.5.
                - 'recall'                   : Recall at cutoff = 0.5.
                - 'auto'                     : Compute all metrics.
                - 'evaluation_data'          : Get the predictions and the
                                               inferred ground truth labels.

        thresholds: list[float]
            List of thresholds to use for precision/recall computation

        verbose: boolean, optional
            When set to true, more status output is generated
            Default: True

        Returns
        -------
        out : dict
            Dictionary of evaluation results where the key is the name of the
            evaluation metric (e.g. precision) and the value is the evaluation
            score.

        See Also
        --------
        create

        Examples
        --------
        .. sourcecode:: python

            >>> metrics = model.evaluate(valid, time_boundary = churn_boundary_oct)
            >>> print metrics
            {'auc'      : 0.6634142545907242,
             'recall'   : 0.6243386243386243,
             'precision': 0.6310160427807486,
             'evaluation_data':
                     +------------+-----------------+-------+
                     | CustomerID |   probability   | label |
                     +------------+-----------------+-------+
                     |   12348    |  0.93458378315  |   1   |
                     |   12361    |  0.437742382288 |   1   |
                     |   12365    |       0.5       |   1   |
                     |   12375    |  0.769197463989 |   0   |
                     |   12380    |  0.339888572693 |   0   |
                     |   12418    |  0.15767210722  |   1   |
                     |   12432    |  0.419652849436 |   0   |
                     |   12434    |  0.88883471489  |   1   |
                     |   12520    | 0.0719764530659 |   1   |
                     |   12546    |  0.949095606804 |   0   |
                     +------------+-----------------+-------+
                     [359 rows x 3 columns]
             'roc_curve':
                    +-----------+-----+-----+-----+-----+
                    | threshold | fpr | tpr |  p  |  n  |
                    +-----------+-----+-----+-----+-----+
                    |    0.0    | 1.0 | 1.0 | 189 | 170 |
                    |   1e-05   | 1.0 | 1.0 | 189 | 170 |
                    |   2e-05   | 1.0 | 1.0 | 189 | 170 |
                    |   3e-05   | 1.0 | 1.0 | 189 | 170 |
                    |   4e-05   | 1.0 | 1.0 | 189 | 170 |
                    |   5e-05   | 1.0 | 1.0 | 189 | 170 |
                    |   6e-05   | 1.0 | 1.0 | 189 | 170 |
                    |   7e-05   | 1.0 | 1.0 | 189 | 170 |
                    |   8e-05   | 1.0 | 1.0 | 189 | 170 |
                    |   9e-05   | 1.0 | 1.0 | 189 | 170 |
                    +-----------+-----+-----+-----+-----+
                    [100001 rows x 5 columns]
             'precision_recall_curve':
                     +---------+----------------+----------------+
                     | cutoffs |   precision    |     recall     |
                     +---------+----------------+----------------+
                     |   0.1   | 0.568181818182 | 0.925925925926 |
                     |   0.25  |  0.6138996139  | 0.84126984127  |
                     |   0.5   | 0.631016042781 | 0.624338624339 |
                     |   0.75  | 0.741935483871 | 0.243386243386 |
                     |   0.9   | 0.533333333333 | 0.042328042328 |
                     +---------+----------------+----------------+
                     [5 rows x 3 columns]
            }
        """
        _mt._get_metric_tracker().track('toolkits.churn_predictor.evaluate')

        data = self._evaluation_prepare(dataset, time_boundary, user_data,
                verbose)
        return self._get_metrics_from_predictions(data, metric, cutoffs)

    @_viewPublic
    def _evaluate_auc(self, prepared_dataset):
        """
        Computes AUC given a dataset prepared by calling the
        ``evaluation_prepare`` method. To make metrics computation more
        efficient, the `evaluation_prepare``
        method is called once, and then metrics can be computed.

        Parameters
        ----------
        prepared_dataset : SFrame
            A dataset prepared by calling ``evaluation_prepare``.

        Returns
        -------
        out : double

        See Also
        --------
        create, gl.evaluation.auc
        """
        if "label" not in prepared_dataset.column_names():
            raise _ToolkitError(
                    "Column 'label' must be present in input data")
        if "probability" not in prepared_dataset.column_names():
            raise _ToolkitError(
                "Column 'probability' must be present in input data")

        return _gl.evaluation.auc(prepared_dataset["label"],
                                  prepared_dataset["probability"])

    @_viewPublic
    def _evaluate_roc_curve(self, prepared_dataset):
        """
        Computes ROC curve given a dataset prepared by calling the
        ``evaluation_prepare`` method. To make metrics computation more
        efficient, the `evaluation_prepare`` method is called once, and then
        metrics can be computed.

        To plot the ROC curve, use the true positive rate on the Y-Axis and the
        false positive rate on the X-Axis.

        Parameters
        ----------
        prepared_dataset : SFrame
            A dataset prepared by calling ``evaluation_prepare``.

        Returns
        -------
        out : SFrame

        See Also
        --------
        create, gl.evaluation.roc_curve
        """
        return _gl.evaluation.roc_curve(prepared_dataset["label"],
                prepared_dataset["probability"])

    @_viewPublic
    def _evaluate_precision(self, prepared_dataset, threshold=0.5):
        """
        Computes the precision given a dataset prepared by calling the
        ``evaluation_prepare`` method. To make metrics computation more
        efficient, the `evaluation_prepare`` method is called once, and then
        metrics can be computed.

        Parameters
        ----------
        prepared_dataset : SFrame
            A dataset prepared by calling ``evaluation_prepare``.

        threshold : double
            The threshold above which a user is predicted as churning.

        Returns
        -------
        out : double
            The precion at the given threshold

        See Also
        --------
        create, gl.evaluation.precision
        """
        predictions = prepared_dataset["probability"] > threshold
        return _gl.evaluation.precision(prepared_dataset["label"],
                predictions)

    @_viewPublic
    def _evaluate_recall(self, prepared_dataset, threshold=0.5):
        """
        Computes the recall given a dataset prepared by calling the
        ``evaluation_prepare`` method. To make metrics computation more
        efficient, the `evaluation_prepare`` method is called once, and then
        metrics can be computed.

        Parameters
        ----------
        prepared_dataset : SFrame
            A dataset prepared by calling ``evaluation_prepare``.

        threshold : double
            The threshold above which a user is predicted as churning.

        Returns
        -------
        out : double
            The recall at the given threshold

        See Also
        --------
        create, gl.evaluation.recall
        """
        predictions = prepared_dataset["probability"] > threshold
        return _gl.evaluation.recall(prepared_dataset["label"],
                predictions)

    @_viewPublic
    def _get_metrics_from_predictions(self,
                data,
                metric,
                cutoffs=[0.1, 0.25, 0.5, 0.75, 0.9]):
        """
        Internal prediction from prepared data.
        """
        # Error checking
        # --------------------------------------------------------------------
        _raise_error_if_not_of_type(metric, [list, str], "metric")
        _raise_error_if_not_of_type(cutoffs, [list], "thresholds")

        _SUPPORTED_METRICS = ['auc', 'precision', 'recall', 'roc_curve',
                'precision_recall_curve']
        if metric == 'auto':
            metric = _SUPPORTED_METRICS
        if type(metric) == list:
            for m in metric:
                _check_categorical_option_type('metric', m, _SUPPORTED_METRICS)
        if type(metric) == str:
            _check_categorical_option_type('metric', metric,
                    _SUPPORTED_METRICS)
            metric = [metric]

        # Evaluate.
        # --------------------------------------------------------------------
        out = {}
        out['evaluation_data'] = data
        if 'auc' in metric:
            out['auc'] = self._evaluate_auc(data)
        if 'roc_curve' in metric:
            out['roc_curve'] = self._evaluate_roc_curve(data)
        if 'precision' in metric:
            out['precision'] = self._evaluate_precision(data, 0.5)
        if 'recall' in metric:
            out['recall'] = self._evaluate_recall(data, 0.5)
        if 'precision_recall_curve' in metric:
            recall = [self._evaluate_recall(data, r) for r in cutoffs]
            precision = [self._evaluate_precision(data, r) for r in cutoffs]
        if 'precision_recall_curve' in metric:
            out['precision_recall_curve'] = _gl.SFrame({
                    'cutoffs': cutoffs,
                    'precision': precision,
                    'recall': recall,
                })
        return out

    def get_activity_baseline(self):
        """
        Get a baseline Churn prediction activity model which uses the time
        since last seen to predict churn.

        Returns
        -------
        out: Model of type ChurnPredictorActivityModel

        Examples
        --------
        .. sourcecode:: python

            >>> activity_model = model.get_activity_baseline()
        """
        _mt._get_metric_tracker().track(
                'toolkits.churn_predictor.get_activity_baseline')
        return ChurnPredictorActivityModel(self.__proxy__)

    def get_feature_importance(self):
        """
        Get the importance of features used by the model.

        The measure of importance of feature X is determined by the sum of
        occurence of X as a branching node in all trees.

        When X is a categorical feature, e.g. "Gender", the index column
        contains the value of the feature, e.g. "M" or "F".  When X is a
        numerical feature, index of X is None.

        Return
        ------
        out : SFrame
            A table with three columns: name, index, count,
            ordered by 'count' in desending order.

        Examples
        --------
        """
        _mt._get_metric_tracker().track(
                'toolkits.churn_predictor.get_feature_importance')
        importance = self.trained_model.get_feature_importance()
        period = _make_period_string(self.time_period)
        boundary = ""

        importance['description'] = _translate_feature_name_to_english(
                             importance, period, boundary)
        return importance

    @property
    def views(self):
        """
        Interactively visualize a :class:`~graphlab.churn_predictor.ChurnPredictor`
        model.

        Once a model has been trained, you can easily visualize the model. There
        are three built-in visualizations to help explore, explain, and evaluate
        the model.

        Examples
        --------
        .. sourcecode:: python

             # Explore predictions
             >>> time_boundary = datetime.datetime(2011, 10, 1)
             >>> view = model.views.explore(train, time_boundary)
             >>> view.show()

             # Explore predictions
             >>> time_boundary = datetime.datetime(2011, 10, 1)
             >>> view = model.views.evaluate(train, time_boundary)
             >>> view.show()

             # Combine explore and evaluate
             >>> view = model.views.overview(time_series, eval_date)
             >>> view.show()

        See Also
        --------
        graphlab.churn_predictor._churn_predictor.ChurnPredictorViews
        """
        return ChurnPredictorViews(self)

#-----------------------------------------------------------------------------
#
#                   Churn predictor activity model.
#
#-----------------------------------------------------------------------------
class ChurnPredictorActivityModel(ChurnPredictor):
    def __init__(self, churn_proxy):
        self.__proxy__ = churn_proxy

        # Get the name of the feature.
        # --------------------------------------------------------------------
        bt_model = churn_proxy.get('trained_model')
        features = _gl.boosted_trees_classifier.BoostedTreesClassifier(
                                                  bt_model).get('features')

        # Extract the feature for user time since seen.
        # --------------------------------------------------------------------
        feature_col = list(filter(lambda x: '||' in x, features))[-1]
        feature = 'user_timesinceseen'

        sf = churn_proxy.get("processed_training_data")
        data = _gl.SFrame({
         'label': sf['__internal_label']
        })
        data[feature_col] = sf[feature_col].apply(
                                 lambda x: {feature: x.get(feature, 0)})

        # Train a logistic regression model.
        # --------------------------------------------------------------------
        lr_model = _gl.logistic_classifier.create(data, 'label',
                validation_set = None, verbose = False)
        self._lr_model = lr_model

    def _get_wrapper(self):
        proxy_wrapper = self.__proxy__._get_wrapper()

        def model_wrapper(unity_proxy):
            model_proxy = proxy_wrapper(unity_proxy)
            return ChurnPredictorActivityModel(model_proxy)
        return model_wrapper

    def get(self, field):
        """
        Return the value of a given field. The list of all queryable fields is
        detailed below, and can be obtained programmatically with the
        :func:`~graphlab.churn_predictor.ChurnPredictor.list_fields`
        method.

        +---------------------------+-------------------------------------------------------------+
        |      Field                | Description                                                 |
        +===========================+=============================================================+
        | churn_period              | Period of in-activity used to define churn.                 |
        +---------------------------+-------------------------------------------------------------+
        | grace_period              | Period within the churn_period where churn is allowed       |
        +---------------------------+-------------------------------------------------------------+
        | categorical_features      | List of features treated as categorical features.           |
        +---------------------------+-------------------------------------------------------------+
        | numerical_features        | List of features treated as numerical features.             |
        +---------------------------+-------------------------------------------------------------+
        | features                  | List of all features used in the model.                     |
        +---------------------------+-------------------------------------------------------------+
        | is_data_aggregated        | Is the input data already in an aggregated format?          |
        +---------------------------+-------------------------------------------------------------+
        | lookback_periods          | Periods to lookback into the past during feature creation.  |
        +---------------------------+-------------------------------------------------------------+
        | model_options             | Additional options provided to the boosted tree classifier. |
        +---------------------------+-------------------------------------------------------------+
        | num_features              | Number of features used while training the model.           |
        +---------------------------+-------------------------------------------------------------+
        | num_observations          | Number of observations in the training data.                |
        +---------------------------+-------------------------------------------------------------+
        | num_users                 | Number of users in the training data.                       |
        +---------------------------+-------------------------------------------------------------+
        | processed_training_data   | Output of the feature engineering steps.                    |
        +---------------------------+-------------------------------------------------------------+
        | time_boundaries           | Churn boundaries used while training the model.             |
        +---------------------------+-------------------------------------------------------------+
        | time_period               | Time-scale at which patterns are learned.                   |
        +---------------------------+-------------------------------------------------------------+
        | user_id                   | Column name with the user id.                               |
        +---------------------------+-------------------------------------------------------------+
        | trained_model             | Trained model used for predictions                          |
        +---------------------------+-------------------------------------------------------------+
        | trained_explanation_model | Trained model used for generating segments/explanations     |
        +---------------------------+-------------------------------------------------------------+

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out
            Value of the requested fields.

        See Also
        --------
        list_fields

        Examples
        --------
        .. sourcecode:: python

            >>> model = graphlab.logistic_classifier.create(data,
            ...                target='is_expensive', features=['bedroom', 'size'])

            # Data computed at the end of feature engineering.
            >>> feature_engineered_data = model.processed_training_data
        """
        _mt._get_metric_tracker().track('toolkits.churn_predictor.get')
        if field == 'trained_model':
            return self._lr_model
        elif field == 'trained_explanation_model':
            return None
        else:
            return super(ChurnPredictorActivityModel, self).get(field)

    def get_activity_baseline(self):
        """
        Get a baseline Churn prediction activity model which uses the time
        since last seen to predict churn.

        Returns
        -------
        out: Model of type ChurnPredictorActivityModel

        Examples
        --------
        .. sourcecode:: python

            >>> activity_model = model.get_activity_baseline()
        """
        return self

    def _internal_predict(self, dataset, time_boundary=None, user_data=None,
            include_leaf_id = False, include_label = False,
            include_bad_users = True, verbose=True):
        """
        Internal predict method that is re-used by the churn report as well
        as a predict functions.

        Parameters
        ----------
        dataset :TimeSeries
            A dataset that has the same columns that were used during training.

        time_boundary : datetime, optional
            Time-boundary strictly after which a prediction for churn/stay is
            made. By default the last timestamp of the `dataset` is used.

            The time boundary is used to compute features as well as make a
            prediction for whether or not the user will churn/stay after the
            time-boundary. All the data present before the boundary will be
            used to compute features, and the data after the boundary will be
            ignored.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column
            with the same name as what is specified by the `user_id` input
            parameter.  `user_data` can provide any amount of additional
            user-specific information. The join performed is an inner join.

        include_leaf_id: bool, optional
            Include leaf_id for predictions? (used by churn report)

        include_label: bool, optional
            Include label for evaluation?

        include_bad_users: bool, optional
            Include users without enough data?

        verbose: boolean, optional
            When set to true, more status output is generated
            Default: True

        """
        _raise_error_if_not_timeseries(dataset, "dataset")
        _raise_error_if_not_of_type(time_boundary,
            [_NoneType, _datetime.datetime], "time_boundary")
        _raise_error_if_not_of_type(user_data, [_NoneType, _gl.SFrame],
                "user_data")
        if include_leaf_id == True:
            raise _ToolkitError("Cannot obtain a churn report from the "
                    "activity model.")

        if time_boundary is None:
            start_time = dataset.max_time
        else:
            start_time = time_boundary
        end_time = start_time + self.get("churn_period")
        _print_with_progress("Making a churn forecast for the time window:",
                                            verbose)
        _print_with_progress(_LINE_MSG, verbose)
        _print_with_progress(" Start : %s" % start_time, verbose)
        _print_with_progress(" End   : %s" % end_time, verbose)
        _print_with_progress(_LINE_MSG, verbose)

        # Call extract features.
        # ---------------------------------------------------------------------
        features_data, bad_users = self.extract_features(dataset,
                time_boundary, user_data, verbose)
        num_bad_users = len(bad_users)

        # Make prediction from the model.
        # ---------------------------------------------------------------------
        user_id = self.user_id
        predictions = _gl.SFrame({
            user_id : features_data[user_id],
        })
        predictions["probability"] =  self._lr_model.predict(features_data,
                output_type = 'probability')

        # Take care of bad users.
        if (len(bad_users) > 0):
            _print_with_progress(
                "Not enough data to make predictions for %s user(s). "\
                                            % len(bad_users), verbose)
        bad_users['probability'] = _gl.SArray.from_const(None,
                num_bad_users).astype(float)

        # Include the label (used for evaluation)
        # -----------------------------------------------------------------
        if include_label:
            predictions['label'] = features_data['__internal_label']
            bad_users['label'] = _gl.SArray.from_const(None,
                    num_bad_users).astype(int)

        if include_bad_users:
            predictions = predictions.append(bad_users)
        return predictions

    def get_feature_importance(self):
        """
        Get the importance of features used by the model. (This feature
        is currently not availiable for this model).
        """
        _mt._get_metric_tracker().track(
                'toolkits.churn_predictor.get_feature_importance')
        raise _ToolkitError("Feature importance is not avaliable for this model")

class ChurnPredictorViews(object):
    def __init__(self, model):
        self.model = model

    def __repr__(self):
        s  = ["Available views for this ChurnPredictorModel"]
        s += [_LINE_MSG]
        s += ["description : Show training statistics for the model."]
        s += ["explore     : Explore the model qualitatively."]
        s += ["evaluate    : Understand model performance quantitatively."]
        s += ["overview    : All of the above, combined into a single UI."]
        return '\n'.join(s)

    def overview(self,
            exploration_set,
            exploration_time,
            validation_set = None,
            validation_time = None,
            user_data = None,
            baseline = None):
        """
        Interactively explore and evaluate a trained churn predictor model.

        Creates a visualization of the performance of this model relative to a
        baseline model, and a visualization of the model that helps explore and
        qualitatively evaluate the churn predicted by the model, deploys both
        to a local ViewServer, and returns the resulting view.

        Parameters
        ----------
        exploration_set: SFrame
            A dataset that has the same columns that were used during training.
            The data set is to create the `explore` view.

        exploration_time: datetime
            Time to use as a boundary for exploration. User activity beyond
            that point will be used to determine whether users have churned. If
            set to None, we use the same ``time_boundary`` as that obtained
            during training.

        validation_set: SFrame (optional)
            A dataset that has the same columns that were used during training.
            The data set is used to create the `evaluate` view.

        evaluation_time: datetime (optional)
            Time to use as a boundary for evaluation. User activity beyond that
            point will be used to determine whether users have churned. If set
            to None, we use ``exploration_time``.

        user_data : SFrame, optional (optional)
            Side information for the users.  This SFrame must have a column
            with the same name as what is specified by the `user_id` input
            parameter.  `user_data` can provide any amount of additional
            user-specific information. The join performed is an inner join.

        baseline: optional
            Add an (optional) baseline model to compare against.

        Returns
        -------
        out : View
            This object can be opened in a web browser with .show().

        .. sourcecode:: python

            >>> view = model.views.overview(time_series, eval_date)
            >>> view.show()
        """
        _mt._get_metric_tracker().track('toolkits.churn_predictor.overview')
        if validation_time is None:
            validation_time = exploration_time

        if validation_set is None:
            validation_set = exploration_set
            num_validation_users = 'num_users'
            num_validation_observations = 'num_observations'
        else:
            validation_data_sf = validation_set.to_sframe()
            num_validation_observations = _precomputed_field(validation_data_sf.num_rows())
            user_id = self.model.get("user_id")
            num_validation_users = _precomputed_field(
                len(validation_data_sf[user_id].unique())
            )

        validation_summary = (
        [
          [
            ('Number of observations', num_validation_observations),
            ('Number of users', num_validation_users),
            ('Time Boundary', _precomputed_field(str(validation_time)))
          ]
        ],
          ["Validation"]
        )
        return _OverviewApp(
            self.description(validation_summary),
            self.explore(
                observation_data=exploration_set,
                time_boundary=exploration_time,
                user_data=user_data),
            self.evaluate(
                observation_data=validation_set,
                time_boundary=validation_time,
                user_data=user_data,
                baseline=baseline),
            title="Churn Predictor View")

    def description(self, extra_info):
        """
        Create a visualization of a description of the model: model type,
        and information about the training data.

        Returns
        -------
        out : View

        .. sourcecode:: python

            >>> view = model.views.description()
            >>> view.show()
        """
        return _model_description(self.model, extra_info = extra_info)

    def evaluate(self, observation_data, time_boundary = None, user_data =
            None, baseline = None):
        """
        Create a view for evaluation for churn prediction.

        Parameters
        ----------
        observation_data: SFrame
            A dataset that has the same columns that were used during training.

        time_boundary: datetime (optional)
            Time to use as a boundary for evaluation. User activity beyond that
            point will be used to determine whether users have churned. If set
            to None, we use the same ``time_boundary`` as that obtained during
            training.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column
            with the same name as what is specified by the `user_id` input
            parameter.  `user_data` can provide any amount of additional
            user-specific information. The join performed is an inner join.

        baseline: optional
            Add an (optional) baseline model to compare against.

        Returns
        -------
        out : View
            This object can be visualized with .show().

        See Also
        --------
        create

        Examples
        --------

        .. sourcecode:: python

            # Create an interactive evaluate view for date in the past
            >>> view = model.views.evaluate(valid, date_in_past)
            >>> view.show()

        """
        _mt._get_metric_tracker().track('toolkits.churn_predictor.evaluate')
        if baseline is not None:
            models = {'churn_predictor': self.model, 'baseline': baseline}
        else:
            baseline = self.model.get_activity_baseline()
            models = {'churn_predictor': self.model, 'baseline': baseline}

        if time_boundary is None:
            time_boundary = max(self.model.time_boundaries)
        return _ChurnPredictorEvaluateView(observation_data,
                  time_boundary, user_data, models)


    def explore(self, observation_data, time_boundary=None, user_data=None):
        """
        Interactively explore predictions made by the churn predictor model.

        Parameters
        ----------
        observation_data: TimeSeries
            A dataset that has the same columns that were used during training.

        time_boundary : datetime, optional
            Time-boundary strictly after which a prediction for churn/stay is
            made. By default the last timestamp of the `dataset` is used.

        user_data : SFrame, optional
            Side information for the users.  This SFrame must have a column
            with the same name as what is specified by the `user_id` input
            parameter.  `user_data` can provide any amount of additional
            user-specific information. The join performed is an inner join.

        verbose: boolean, optional
            When set to true, more status output is generated
            Default: True

        Returns
        -------
        out : SFrame
            An SFrame with model predictions. A prediction of 100% means the
            user is likely to stay, whereas a prediction of 0% means the
            user is likely to leave (churn).

        See Also
        --------
        create

        Returns
        -------
        out : View
            This object can be visualized with .show().

        Examples
        --------

        .. sourcecode:: python

            # Create an interactive evaluation view for a future forecast.
            >>> view = model.views.explore(time_series)
            >>> view.show()
        """
        _mt._get_metric_tracker().track('toolkits.churn_predictor.explore')
        if time_boundary is None:
            time_boundary = observation_data.max_time
        return _ChurnPredictorExploreView(self.model, observation_data,
			                time_boundary, user_data)
