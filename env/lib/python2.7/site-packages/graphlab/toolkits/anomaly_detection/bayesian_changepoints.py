"""
Class definition and utilities for the time series changepoint
detection tool.
"""

import time as _time
import graphlab as _gl
import graphlab.connect as _mt
from graphlab.toolkits._model import SDKModel as _SDKModel
import graphlab.toolkits._internal_utils as _tkutl
from graphlab.toolkits._private_utils import _summarize_accessible_fields
from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits._model import _get_default_options_wrapper
import datetime as _dt
import logging as _logging

get_default_options = _get_default_options_wrapper('_BayesianOnlineChangepoint',
    '_BayesianOnlineChangepoint', '_BayesianOnlineChangepoint', True)



def create(dataset, feature=None, expected_runlength=250, lag=7):
    """
    Create a `BayesianChangepointsModel`. The changepoint detection
    calculates where there is a shift in mean or variance in a univariate
    timeseries. This model calculates a probability that a given point is
    changepoint, given the data up to the point. The BayesianChangepointsModel
    works with either TimeSeries, SArray, or SFrame inputs.

    The model created by this function contains a table `scores` that contains
    the computed anomaly scores. The type of `scores` matches the type of the
    input `dataset`, and the table contains 4 columns:

        - *row id/time*: ID of the corresponding row in the input `dataset`. If
          `dataset` is an SFrame, this is the row numbers of the input data; if
          `dataset` is a TimeSeries, it is the index of the time series.

        - *changepoint score*: The probability that the given point is a
           changepoint. This value is in a range between 0 and 1.

        - *value*: input data. The name of this column matches the input
          `feature`.

        - *model update time*: time the model was updated. This is particularly
          useful if the `window_size` is larger than the number of rows in the
          input datasets, because the `scores` table has results from several
          updates.

    Note that any `None` values in dataset will have `changepoint_score` of
    `None`, and will be ignored in subsequent changepoint probability
    calculation.

    Parameters
    ----------
    dataset : SFrame, SArray, or TimeSeries
        Input data. The column named by 'feature' will be extracted for
        modeling.

    feature : str, optional
        Name of the column to model. Any data provided to the model in this
        function or with `BayesianChangepointsModel.update` must have a column
        with this name, unless the datasets are in SArray form.

    expected_runlength: int or float, optional
       The a priori expected number of samples between changepoints.
       This helps condition the model. Note that this parameter must be set
       to a value larger than 0.

    lag: int, optional
        The model waits `lag` samples before evaluating the probability of a
        change happening `lag` samples prior. This is useful because it can be
        difficult to evaluate a change after a single sample of a new
        distribution.

        Note that this causes the last `lag` to not have enough data to
        evaluate changepoint scores, so they are filled with 'None' values.
        Also note that this value cannot be larger than 100, due to only
        keeping the previous 100 points in memory.The minimum lag is 0, which
        allows immediate detection of changepoints, but with less certainty.

    Returns
    -------
    out : BayesianChangepointsModel

    See Also
    --------
    MovingZScoreModel, graphlab.TimeSeries, local_outlier_factor

    References
    ----------
    - The model implemented is desribed in `'Bayesian Online Changepoint Prediction'
      by Ryan Adams, <http://arxiv.org/pdf/0710.3742v1.pdf>`_.

    Examples
    --------
    >>> sf = graphlab.SFrame({'series': [100]*25 + [200]*25})
    >>> model = graphlab.anomaly_detection.bayesian_changepoints.create(sf,
    ...                                                         lag=5,
    ...                                                         feature='series')
    >>> model['scores'][24:28].print_rows(max_column_width=20)
    +--------+-------------------+--------+---------------------+
    | row_id | changepoint_score | series |  model_update_time  |
    +--------+-------------------+--------+---------------------+
    |   24   |   0.136735367681  |  100   | 2016-01-27 14:02... |
    |   25   |   0.831430606595  |  200   | 2016-01-27 14:02... |
    |   26   | 0.000347138442071 |  200   | 2016-01-27 14:02... |
    |   27   | 3.40869782692e-05 |  200   | 2016-01-27 14:02... |
    +--------+-------------------+--------+---------------------+
    [4 rows x 4 columns]
    """
    _mt._get_metric_tracker().track('{}.create'.format(__name__))

    start_time = _time.time()
    logger = _logging.getLogger(__name__)

    ## Validate required inputs by themselves.
    if not isinstance(dataset, (_gl.SFrame, _gl.TimeSeries)):
        raise TypeError("Input 'dataset' must be an SFrame or TimeSeries.")

    if len(dataset) < 1:
        raise _ToolkitError("Input 'dataset' is empty.")

    if feature is not None and not isinstance(feature, str):
        raise TypeError("Input 'feature' must be a string if specified.")

    if not isinstance(lag, int):
        raise TypeError("Input 'lag' must be an integer if specified.")

    if lag > 100 or lag < 0:
        raise ValueError("Input 'lag' cannot be greater than 100 or less than 0")

    if type(expected_runlength) not in (int, float):
        raise TypeError("'expected_runlength' must be either an integer or float")

    if expected_runlength < 1:
        raise ValueError("Input 'expected_runlength' must be greater than 0.")


    ## Determine the feature name if left unspecified.
    column_names = dataset.column_names() if isinstance(dataset, _gl.SFrame) \
        else dataset.value_col_names

    if feature is None:
        if len(column_names) == 1:
            feature = column_names[0]
        else:
            raise _ToolkitError("If the 'input' dataset has multiple " +
                                "columns, a 'feature' column name must be " +
                                "specified.")


    ## Extract the specified feature as an SArray.
    try:
        series = dataset[feature]
    except:
        raise _ToolkitError("The specified feature could not be found " +
                            "in the input 'dataset'.")

    ## Validate the type of the feature.
    if not series.dtype() in [int, float]:
        raise ValueError("The values in the specified feature must be " +
                         "integers or floats.")

    ## Initialize options
    opts = {}

    opts['expected_runlength'] = expected_runlength
    opts['lag']  = lag
    opts['feature'] = feature

    ## Create SDK proxy
    proxy = _gl.extensions._BayesianOnlineChangepoint()
    proxy.init_changepoint_detector(opts, False, series.dropna()[0])

    ## Construct python model from proxy
    model = BayesianChangepointsModel(proxy)

    ## Construct scores SFrame from calculated changepoints
    scores = _gl.SFrame()
    scores[feature] = series
    changepoints = model.__proxy__.calculate_changepoints(series)

    ## Append None's at the end, where there hasn't been enough data to determine
    ## whether there was a changepoint
    changepoints = changepoints.append(_gl.SArray([None]* (len(scores)-len(changepoints))))
    scores['changepoint_score'] = changepoints
    scores['model_update_time'] = _dt.datetime.now()

    scores = scores[['changepoint_score', # reorder the columns
                     feature,
                     'model_update_time']]

    #Add row_id to SFrame
    if isinstance(dataset, _gl.SFrame):
        if feature != 'row_id':
            scores = scores.add_row_number('row_id')
        else:
            logger.warning("Feature name is 'row_id', so the " +
                           "index in the model's 'scores' SFrame " +
                           "is called '_row_id'.")
            scores = scores.add_row_number('_row_id')

    ## Add index to timeseries
    if isinstance(dataset, _gl.TimeSeries):
        scores[dataset.index_col_name] = dataset[dataset.index_col_name]

    dataset_type = 'TimeSeries' if isinstance(dataset, _gl.TimeSeries) else 'SFrame'

    # Set up the model.
    state = {
        'dataset_type': dataset_type,
        'num_examples': len(dataset),
        'training_time': _time.time() - start_time}

    if isinstance(dataset, _gl.TimeSeries):
        model.__proxy__.set_index_col_name(dataset.index_col_name)
        model.__proxy__.set_state_sframe(scores, state)
    else:
        model.__proxy__.set_state_sframe(scores, state)

    return model


class BayesianChangepointsModel(_SDKModel):
    """
    Identify changepoints in a  univariate time series.
    A created Changepoint contains a regime id for each point in the current d
    ataset, and can predict changepoints for new
    data instances with the `update` method.

    Unlike most models in GraphLab Create, there are two ways to construct a
    ``BayesianChangepointsModel``: `graphlab.bayesian_changepoints.changepoint_model.create`
    and the `update` method of an existing model. Please note that
    BayesianChangepointsModel instances are essentially immutable: the `update`
    method does not change the state of an existing model.

    This model should not be constructed directly.
    """


    def __init__(self, model_proxy):
        self.__proxy__ = model_proxy

    def _get_wrapper(self):
        proxy_wrapper = self.__proxy__._get_wrapper()

        def model_wrapper(unity_proxy):
            model_proxy = proxy_wrapper(unity_proxy)
            return BayesianChangepointsModel(model_proxy)
        return model_wrapper


    def update(self, dataset):
        """
        Create a new BayesianChangepointsModel using the same parameters, but
        an updated dataset. Knowledge about the data is retained from the
        previous model, and it is assumed the data is a continuation of the
        previous models data.

        Parameters
        ----------
        dataset : SFrame, SArray, or TimeSeries
            New data to use for an updated changepoint detection model. The
            type of the input 'dataset' must match the type of the data already
            in the model (if the model has data already).

        Returns
        -------
        out : BayesianChangepointsModel
            A *new* BayesianChangepointsModel, with an updated dataset and
            changepoint scores for the updated dataset. The `scores` field of
            the new model has the same schema as the `scores` field of the
            existing model. The last `lag` fields are prepended to the data,
            though, because there is now enough data to evaluate their
            changepoint probability.

        See Also
        --------
        create

        Examples
        --------
        >>> sf = graphlab.SFrame({'series': [100]*25})
        >>> model = graphlab.anomaly_detection.bayesian_changepoints.create(sf,
        ...                                                         lag=5,
        ...                                                         feature='series')
        >>> sf2 = graphlab.SFrame({'series': [200]*25})
        >>> model2 = model.update(sf2)
        >>> model2['scores'].print_rows(max_column_width=20)
        +-------------------+--------+---------------------+
        | changepoint_score | series |  model_update_time  |
        +-------------------+--------+---------------------+
        |   0.831430606595  |  200   | 2016-01-27 14:06... |
        | 0.000347138442071 |  200   | 2016-01-27 14:06... |
        | 3.40869782692e-05 |  200   | 2016-01-27 14:06... |
        | 1.40792637711e-05 |  200   | 2016-01-27 14:06... |
        | 7.50780005726e-06 |  200   | 2016-01-27 14:06... |
        | 4.49582032092e-06 |  200   | 2016-01-27 14:06... |
        | 2.90328065455e-06 |  200   | 2016-01-27 14:06... |
        | 1.98060675567e-06 |  200   | 2016-01-27 14:06... |
        | 1.40930691121e-06 |  200   | 2016-01-27 14:06... |
        | 1.03700199168e-06 |  200   | 2016-01-27 14:06... |
        +-------------------+--------+---------------------+
        [25 rows x 3 columns]
        """
        start_time = _time.time()
        _mt._get_metric_tracker().track(
                              'toolkit.anomaly_detection.bayesian_changepoints.update')
        logger = _logging.getLogger(__name__)



        ## Validate the new dataset
        if not isinstance(dataset, (_gl.SFrame, _gl.TimeSeries)):
            raise TypeError("Input 'dataset' must be an SFrame or TimeSeries.")

        if len(dataset) < 1:
            raise TypeError("Input 'dataset' is empty.")

        if ((self.get('dataset_type') == 'TimeSeries' and not isinstance(dataset, _gl.TimeSeries)) or
            (self.get('dataset_type') == 'SFrame' and not isinstance(dataset, _gl.SFrame))):

            raise TypeError("New input 'dataset' must have the same type " +
                            "as the data already in the model.")


        ## TimeSeries-specific dataset validation
            ## Make the sure new data occurs *after* the existing data.
        scores = self.get('scores')

        if isinstance(dataset, _gl.TimeSeries):
            first_new_timestamp = dataset[0][dataset.index_col_name]
            last_old_timestamp = scores[-1][scores.index_col_name]

            if first_new_timestamp < last_old_timestamp:
                raise _ToolkitError("The new dataset has data with " +
                                    "earlier timestamps than the existing " +
                                    "dataset. Please ensure that new data " +
                                    "occurs after existing data.")


        ## Extract the feature from the new dataset and validate it.
        feature = self.get('feature')

        try:
            series = dataset[feature]
        except:
            raise _ToolkitError("The feature specified by the original " +
                                "model could not be found in the input " +
                                "'dataset'.")

        if not series.dtype() in [int, float]:
            raise ValueError("The values in the specified feature must be " +
                             "integers or floats.")


        ## Create a new model initialize it.
        new_state = {k: self.get(k)
            for k in ['dataset_type']}

        opts = self.__proxy__.get_most_likely_hyperparams()

        proxy = _gl.extensions._BayesianOnlineChangepoint()

        ## Initialize new model w/state from old model. This allows for
        ## detection changepoints using knowledge learned previously, and
        ## also to detect changepoint probabilites for points which didn't yet
        ## have `lag` points following them.
        proxy.init_changepoint_detector(opts, True, self.get('scores')[feature].dropna()[0])

        new_model = BayesianChangepointsModel(proxy)


        ## Once again, calculate changepoints with information known
        ## from model creation. Prepend `lag` points from previous dataset,
        ## we now have enough information to check if they were changepoints
        lag = self.get('lag')

        ## If `lag` is greater than 0, we want to prepend the points that we
        ## couldn't find a changepoint score before due to not enough data.
        ## These are the laat `lag` non-None points.
        if lag > 0:
            ## Grab previous scores
            if isinstance(dataset, _gl.SFrame):
                old_scores = self.get('scores')[[feature, 'model_update_time']]
            else:
                old_scores = self.get('scores').to_sframe()[[feature, 'model_update_time']]
            # Copy SFrame and select only the feature column
            prepend_index_calc_temp_sf = old_scores[[feature]]
            #Rename, incase feature column is 'id'
            prepend_index_calc_temp_sf.rename({feature : 'series'})
            # Identify the last `lag` points that are non-None
            prepend_index_calc_temp_sf = prepend_index_calc_temp_sf.add_row_number()
            prepend_index_calc_temp_sf = prepend_index_calc_temp_sf.dropna()
            # If `lag` is longer than scores, just take all previous points
            if lag >= len(prepend_index_calc_temp_sf):
                prepend_index = 0
            else:
                prepend_index = prepend_index_calc_temp_sf['id'][-(lag + 1)] + 1
            old_scores = old_scores[prepend_index:]
        ## Otherwise, we don't prepend anything, so the index can be the input
        ## data length
        else:
            prepend_index = len(series)

        ## Calculate changepoints

        scores = _gl.SFrame()
        scores[feature] = series
        changepoints =new_model.__proxy__.calculate_changepoints(series)
        scores['model_update_time'] = _dt.datetime.now()
        if lag > 0:
            scores = old_scores.append(scores)
        changepoints = changepoints.append(_gl.SArray([None]* (len(scores)-len(changepoints))))
        scores['changepoint_score'] = changepoints
        scores = scores[['changepoint_score', # reorder the columns
                     feature,
                     'model_update_time']]

        ## Add row_id to SFrame
        if isinstance(dataset, _gl.SFrame):
            if feature != 'row_id':
                scores = scores.add_row_number('row_id')
            else:
                logger.warning("Feature name is 'row_id', so the " +
                               "index in the model's 'scores' SFrame " +
                               "is called '_row_id'.")
                scores = scores.add_row_number('_row_id')



        ## Finalize and return the model.
        new_state['num_examples'] = len(scores)
        new_state['training_time'] = _time.time() - start_time

        ## If time-series index name has changed, rename old_timeseries
        ## index name
        if isinstance(dataset, _gl.TimeSeries):
            old_index_col_name = self.__proxy__.get_index_col_name();
            old_timeseries = self.get('scores')
            if dataset.index_col_name != old_index_col_name:
                old_timeseries = old_timeseries.rename(
                           {old_index_col_name: dataset.index_col_name})

                logger.warning("The new dataset's index column name " +
                               "does not match the existing index " +
                               "column name. The new name is used in " +
                               "the new model.")

            ## In model creation, the last `lag` points cannot be
            ## evaluated for changepoint probability. Now, there's more data,
            ## so that data is prepended.
            new_index = old_timeseries[dataset.index_col_name][prepend_index:].append(dataset[dataset.index_col_name])
            scores[dataset.index_col_name] = new_index
            new_model.__proxy__.set_index_col_name(dataset.index_col_name)
            new_model.__proxy__.set_state_sframe(scores, new_state)
        else:
            new_model.__proxy__.set_state_sframe(scores, new_state)


        return new_model

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """
        width = 40
        key_str = "{:<{}}: {}"

        sections, section_titles = self._get_summary_struct()
        accessible_fields = {
            "scores": "Changepoint score for each row in the input dataset."}

        out = _tkutl._toolkit_repr_print(self, sections, section_titles,
                                         width=width)
        out2 = _summarize_accessible_fields(accessible_fields, width=width)
        return out + "\n" + out2

    def _get_summary_struct(self):
        """
        Returns a structured description of the model, including (where
        relevant) the schema of the training data, description of the training
        data, training statistics, and model hyperparameters.

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
            ('Number of examples', 'num_examples'),
            ('Feature name', 'feature'),
            ('Expected run length', 'expected_runlength'),
            ('Lag observations','lag')
            ]

        training_fields = [
            ('Total training time (seconds)', 'training_time')
            ]

        section_titles = ['Schema', 'Training summary']
        return([model_fields, training_fields], section_titles)

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current BayesianChangepointsModel instance.
        """
        return self.__proxy__.get_current_options()

    def get(self, field):
        """
        Return the value contained in the model's ``field``.

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out
            Value of the requested field.

        See Also
        --------
        list_fields
        """
        _mt._get_metric_tracker().track(
            'toolkits.anomaly_detection.bayesian_changepoints.get')
        if field == "scores" and self.__proxy__.get('dataset_type') == 'TimeSeries':
            ts = self.__proxy__.get('scores')
            return _gl.TimeSeries(ts, index = self.__proxy__.get_index_col_name())
        else:
            return self.__proxy__.get(field)

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
        >>> fields = model.list_fields()
        """
        _mt._get_metric_tracker().track(
            'toolkits.anomaly_detection.bayesian_changepoints.list_fields')
        return self.__proxy__.list_fields()


