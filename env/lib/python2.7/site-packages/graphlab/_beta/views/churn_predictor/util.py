import datetime as _datetime


def _get_churn_predictor_model_info(model, time_boundary):
    """
    Utility function to get the info about a model.
    """
    forecast_start = time_boundary
    forecast_end = forecast_start + model.churn_period
    model_info_fields = ['categorical_features',
                         'churn_period',
                         'features',
                         'is_data_aggregated',
                         'lookback_periods',
                         'model_options',
                         'num_features',
                         'num_observations',
                         'num_users',
                         'numerical_features',
                         'time_boundaries',
                         'time_period',
                         'user_id']
    info = {
       'model_name' : model.__class__.__name__,
       'forecast_start' : forecast_start,
       'forecast_end'   : forecast_start + model.churn_period
    }
    for key in model_info_fields:
        info[key] = model[key]

    # Cast datetime to string.
    for key in info:
        out = info[key]
        if type(out) == _datetime.timedelta:
            out = out.total_seconds()
        if type(out) == _datetime.datetime:
            out = str(out)
        if key == 'time_boundaries':
            out = map(str, out)
        info[key] = out
    return info
