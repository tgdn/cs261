"""
Class definition and utilities for the Moving Z-Score anomaly detection tool.
"""

import copy as _copy
import time as _time
import logging as _logging
import datetime as _dt
import graphlab as _gl
import graphlab.connect as _mt
from graphlab.toolkits._model import CustomModel as _CustomModel
import graphlab.toolkits._internal_utils as _tkutl
from graphlab.toolkits._private_utils import _summarize_accessible_fields
from graphlab.toolkits._main import ToolkitError as _ToolkitError

from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy


def get_default_options():
    """
    Information about moving Z-score parameters.

    Returns
    -------
    out : SFrame
        Each row in the output SFrames correspond to a parameter, and includes
        columns for default values, lower and upper bounds, description, and
        type.
    """
    out = _gl.SFrame({
        'name': ['window_size', 'feature', 'min_observations', 'verbose'],
        'default_value': ['None', 'None', 'None', 'True'],
            'parameter_type': ['int', 'str', 'int', 'bool'],
        'lower_bound': ['1', 'None', '1', 'False'],
        'upper_bound': ['None', 'None', 'None', 'True'],
        'description': ['Number of observations in the moving window.',
                        'Name of the column to use as a feature.',
                        'Minimum observations in the moving window to compute a Z-score.',
                        'Progress printing flag.']})
    return out


def _moving_z_score(series, window_size, min_observations=None):
    """
    Compute a Z-score for each point in a graphlab.SArray 'series' based on the
    mean and standard deviation in a moving window. Note that `window_size`
    observations must be present for a value to be computed. As a result, the
    output average and moving Z-score for the first `window_size` entries are
    `None`.

    Parameters
    ----------
    series : graphlab.SArray
        Input dataset.

    window_size : int
        Size of the moving window.

    min_observations : int, optional
        Minimum number of non-missing observations in the moving window
        required to compute the moving Z-score. If unspecified, the entire
        moving window preceding an observation must not contain any missing
        values in order for the observation to get an anomaly score.

    Returns
    -------
    moving_average, moving_zscore : graphlab.SArray
        Moving average and Z-score of the input series. Two separate SArrays
        are returned.

    sufficient_data : bool
        If True, there are enough observations to compute at least one moving
        Z-score.
    """

    ## How many observations do we really need?
    if min_observations is None:
        num_required_obs = window_size
    else:
        num_required_obs = min(window_size, min_observations)

    ## If there enough observations compute the moving Z-score. Note that the
    #  series needs to contain at least 1 *more* observation than the minimum,
    #  because the moving window goes from t - window_size to t - 1.
    if len(series) > num_required_obs:
        moving_average = series.rolling_mean(window_start=window_size * -1,
                                             window_end=-1,
                                             min_observations=min_observations)

        moving_std = series.rolling_stdv(window_start=window_size * -1,
                                         window_end=-1,
                                         min_observations=min_observations)

        moving_zscore = (series - moving_average) / moving_std
        sufficient_data = True

    ## If there isn't enough data, manually construct an output of all None's.
    else:
        moving_average = _gl.SArray.from_const(None, len(series))
        moving_zscore = _gl.SArray.from_const(None, len(series))
        sufficient_data = False

    return moving_average, moving_zscore, sufficient_data


def create(dataset, window_size, feature=None, min_observations=None,
           verbose=True):
    """
    Create a :class:`MovingZScoreModel` model. This model fits a moving average
    to a univariate time series and identifies points that are far from the
    fitted curve. The MovingZScoreModel works with either TimeSeries or SFrame
    inputs. A uniform sampling rate is assumed and the data window must be
    defined in terms of number of observations.

    This model differs from other GraphLab Create models in that it can be
    created from an existing `MovingZSCoreModel`. To create a new model in this
    fashion, use the existing model's `update` method.

    The model created by this function contains a table `scores` that contains
    the computed anomaly scores. The type of `scores` matches the type of the
    input `dataset`, and the table contains 5 columns:

        - *row id/time*: ID of the corresponding row in the input `dataset`. If
          `dataset` is an SFrame, this is the row numbers of the input data; if
          `dataset` is a TimeSeries, it is the index of the time series.

        - *anomaly score*: absolute value of the moving Z-score. A score of 0
          indicates the value is identical to the moving average. The higher
          the score, the more likely a point is to be an anomaly.

        - *value*: input data. The name of this column matches the input
          `feature`.

        - *moving average*: moving average of each point's preceding
          `window_size` values.

        - *model update time*: time the model was updated. This is particularly
          useful if the `window_size` is larger than the number of rows in the
          input datasets, because the `scores` table has results from several
          updates.

    Parameters
    ----------
    dataset : SFrame or TimeSeries
        Input data. The column named by the 'feature' parameter will be
        extracted for modeling.

    window_size : int
        Length of the time window to use for defining the moving z-score value,
        in terms of number of observations.

    feature : str, optional
        Name of the column to model. Any data provided to the model with either
        the `create` or `update` functions must have a column with this name.
        The feature name is not necessary if `dataset` is an SFrame with a
        single column or a TimeSeries with a single value column; it can be
        determined automatically in this case.

    min_observations : int, optional
        Minimum number of non-missing observations in the moving window
        required to compute the moving Z-score. If unspecified, the entire
        moving window preceding an observation must not contain any missing
        values in order for the observation to get an anomaly score.

    verbose : bool, optional
        If True, print progress updates and model details.

    Returns
    -------
    out : MovingZScoreModel
        A trained :class:`MovingZScoreModel`, which contains a table called
        `scores` that includes the anomaly score for each input data point. The
        type of the `scores` table matches the type of the input `dataset`.

    See Also
    --------
    MovingZScoreModel, MovingZScoreModel.update

    Notes
    -----
    - The moving Z-score for a data point :math:`x_t` is simply the value of
      :math:`x_t` standardized by subtracting the moving mean just prior to
      time :math:`t` and dividing by the moving standard deviation just prior
      to :math:`t`. Suppose :math:`w` stands for the `window_size` in terms of
      the number of observations. Then the moving Z-score is:

      .. math:: z(x_t) = \\frac{x_t - \\bar{x}_t}{s_t}

      where the moving average is:

      .. math:: \\bar{x}_t = (1/w) \sum_{i=t-w}^{t-1} x_i

      and the moving standard deviation is:

      .. math:: s_t = \sqrt{(1/w) \sum_{i=t-w}^{t-1} (x_i - \\bar{x}_t)^2}

    - The moving Z-score at points within `window_size` observations of the
      beginning of a series are not defined, because there are insufficient
      points to compute the moving average and moving standard deviation. This
      is represented by missing values.

    - Missing values in the input dataset are assigned missing values ('None')
      for their anomaly scores as well.

    - If there is no variation in the values preceding a given observation, the
      moving Z-score can be infinite or undefined. If the given observation is
      equal to the moving average, the anomaly score is coded as 'nan'; if the
      observation is *not* equal to the moving average, the anomaly score is
      'inf'.

    Examples
    --------
    >>> sf = graphlab.SFrame({'year': [2007, 2007, 2008, 2009, 2010, 2010],
    ...                       'value': [12.2, 11.7, 12.5, 21.4, 10.8, 11.2]})
    >>> model = graphlab.anomaly_detection.moving_zscore.create(sf,
    ...                                                         window_size=3,
    ...                                                         feature='value')
    >>> model['scores'].print_rows(max_column_width=20)
    +--------+----------------+-------+----------------+---------------------+
    | row_id | anomaly_score  | value | moving_average |  model_update_time  |
    +--------+----------------+-------+----------------+---------------------+
    |   0    |      None      |  12.2 |      None      | 2016-01-04 16:55... |
    |   1    |      None      |  11.7 |      None      | 2016-01-04 16:55... |
    |   2    |      None      |  12.5 |      None      | 2016-01-04 16:55... |
    |   3    | 28.0822407386  |  21.4 | 12.1333333333  | 2016-01-04 16:55... |
    |   4    | 1.00086199482  |  10.8 |      15.2      | 2016-01-04 16:55... |
    |   5    | 0.795990414837 |  11.2 |      14.9      | 2016-01-04 16:55... |
    +--------+----------------+-------+----------------+---------------------+
    [6 rows x 5 columns]
    """

    _mt._get_metric_tracker().track(
                              'toolkit.anomaly_detection.moving_zscore.create')

    start_time = _time.time()
    logger = _logging.getLogger(__name__)


    ## Validate required inputs by themselves.
    if not isinstance(dataset, (_gl.SFrame, _gl.TimeSeries)):
        raise TypeError("Input 'dataset' must be an SFrame or TimeSeries.")

    if len(dataset) < 1:
        raise _ToolkitError("Input 'dataset' is empty.")

    if not isinstance(window_size, int):
        raise TypeError("Input 'window_size' must be an integer.")

    if window_size < 1:
        raise ValueError("Input 'window_size' must greater than or " +
                         "equal to 1.")

    if feature is not None and not isinstance(feature, str):
        raise TypeError("Input 'feature' must be a string if specified.")

    if min_observations is not None:
        if not isinstance(min_observations, int):
            raise TypeError("If specified, input 'min_observations' must " +
                            "be a positive integer.")

        if min_observations < 1:
            raise ValueError("If specified, input 'min_observations' must " +
                             "be a positive integer.")

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


    ## Compute the moving average, Z-score, and a final anomaly score. For all
    #  anomaly detectcion models, the final score should be in the range [0,
    #  \infty], with higher values indicating more outlier-ness.
    moving_average, moving_zscore, sufficient_data = \
        _moving_z_score(series, window_size, min_observations)

    anomaly_score = abs(moving_zscore)

    if not sufficient_data:
        logger.warning("The number of observations is smaller than " +
                       "the minimum number needed to compute a " +
                       "moving Z-score, so all anomaly scores are 'None'. " +
                       "Consider adding more data with the model's `update` " +
                       "method, or reducing the `window_size` or " +
                       "`min_observations` parameters.")

    ## Format the results.
    scores = _gl.SFrame({feature: series,
                         'moving_average': moving_average,
                         'anomaly_score': anomaly_score})
    scores['model_update_time'] = _dt.datetime.now()

    scores = scores[['anomaly_score', # reorder the columns
                     feature,
                     'moving_average',
                     'model_update_time']]

    if isinstance(dataset, _gl.SFrame):
        if feature != 'row_id':
            scores = scores.add_row_number('row_id')
        else:
            logger.warning("Feature name is 'row_id', so the " +
                           "index in the model's 'scores' SFrame " +
                           "is called '_row_id'.")
            scores = scores.add_row_number('_row_id')

    if isinstance(dataset, _gl.TimeSeries):
        scores[dataset.index_col_name] = dataset[dataset.index_col_name]
        scores = _gl.TimeSeries(scores, index=dataset.index_col_name)

    dataset_type = 'TimeSeries' if isinstance(dataset, _gl.TimeSeries) else 'SFrame'

    ## Set up the model.
    state = {
        'dataset_type': dataset_type,
        'verbose': verbose,
        'window_size': window_size,
        'min_observations': min_observations,
        'num_examples': len(dataset),
        'feature': feature,
        'training_time': _time.time() - start_time,
        'scores': scores}

    model = MovingZScoreModel(state)
    return model


class MovingZScoreModel(_CustomModel, _ProxyBasedModel):
    """
    Identify anomalies in univariate series. A created instance of the
    MovingZScoreModel contains an anomaly score for each point in the current
    dataset, and can predict anomaly scores for new data instances with the
    `update` method. The anomaly score for this model is based on how far each
    data point is from its moving average, measured in terms of the number of
    moving standard deviations.

    Unlike most models in GraphLab Create, there are two ways to construct a
    ``MovingZScoreModel``: `graphlab.anomaly_detection.moving_zscore.create`
    and the `update` method of an existing model. Please note that
    MovingZScoreModels are essentially immutable: the `update` method does not
    change the state of an existing model.

    This model should not be constructed directly.
    """

    _PYTHON_MOVING_ZSCORE_VERSION = 1

    def __init__(self, state):
        self.__proxy__ = _PythonProxy(state)

    def update(self, dataset, window_size=None, min_observations=None,
               verbose=True):
        """
        Create a new `MovingZScoreModel` with a new dataset. The `window_size`
        and `min_observations` parameters can also be updated with this method.

        The new model contains anomaly scores for each observation in the new
        `dataset`. In addition, the last `window_size` rows of the existing
        model's data and anomaly scores are prepended, for continuity and to
        show how the anomaly score is computed for the first few rows of the
        new `dataset`.

        Parameters
        ----------
        dataset : SFrame or TimeSeries
            New data to use for updating the model. The type of the input
            'dataset' must match the type of the data already in the model (if
            the model has data already).

        window_size : int, optional
            Length of the time window to use for defining the moving z-score
            value, in terms of number of observations. The window size will be
            the same as the current model's window size if a new window is not
            specified.

        min_observations : int, optional
            Minimum number of non-missing observations in the moving window
            required to compute the moving Z-score. If unspecified, the entire
            moving window preceding an observation must not contain any missing
            values in order for the observation to get an anomaly score. This
            parameter will be the same as the current model's value if not
            specified.

        verbose : bool, optional
            If True, print progress updates and model details.

        Returns
        -------
        out : MovingZScoreModel
            A *new* MovingZScoreModel, with an updated dataset and anomaly
            scores for the updated dataset. The `scores` field of the new model
            has the same schema as the `scores` field of the existing model,
            but data prepended from the existing results have a row ID of
            'None'.

        See Also
        --------
        create

        Examples
        --------
        >>> sf = graphlab.SFrame({'year': [2007, 2007, 2008, 2009, 2010, 2010],
        ...                       'value': [12.2, 11.7, 12.5, 21.4, 10.8, 11.2]})
        >>> model = graphlab.anomaly_detection.moving_zscore.create(sf,
        ...                                                         window_size=3,
        ...                                                         feature='value')
        ...
        >>> sf2 = graphlab.SFrame({'year': [2010, 2011, 2012, 2013],
        ...                        'value': [18.4, 12.1, 12.0, 3.6]})
        >>> model2 = model.update(sf2)
        >>> model2['scores'].print_rows(max_column_width=20)
        +--------+----------------+-------+----------------+---------------------+
        | row_id | anomaly_score  | value | moving_average |  model_update_time  |
        +--------+----------------+-------+----------------+---------------------+
        |  None  | 28.0822407386  |  21.4 | 12.1333333333  | 2016-01-04 16:58... |
        |  None  | 1.00086199482  |  10.8 |      15.2      | 2016-01-04 16:58... |
        |  None  | 0.795990414837 |  11.2 |      14.9      | 2016-01-04 16:58... |
        |   0    | 0.801849542822 |  18.4 | 14.4666666667  | 2016-01-04 16:58... |
        |   1    | 0.391346818515 |  12.1 | 13.4666666667  | 2016-01-04 16:58... |
        |   2    | 0.593171014002 |  12.0 |      13.9      | 2016-01-04 16:58... |
        |   3    | 3.52963789428  |  3.6  | 14.1666666667  | 2016-01-04 16:58... |
        +--------+----------------+-------+----------------+---------------------+
        [7 rows x 5 columns]
        """
        start_time = _time.time()
        _mt._get_metric_tracker().track(
                              'toolkit.anomaly_detection.moving_zscore.update')
        logger = _logging.getLogger(__name__)


        ## Validate the new dataset
        if not isinstance(dataset, (_gl.SFrame, _gl.TimeSeries)):
            raise TypeError("Input 'dataset' must be an SFrame or TimeSeries.")

        if len(dataset) < 1:
            raise TypeError("Input 'dataset' is empty.")

        if ((self.__proxy__['dataset_type'] == 'TimeSeries' and not isinstance(dataset, _gl.TimeSeries)) or
            (self.__proxy__['dataset_type'] == 'SFrame' and not isinstance(dataset, _gl.SFrame))):

            raise TypeError("New input 'dataset' must have the same type " +
                            "as the data already in the model.")

        ## Validate the new window size (if there is one), and figure out what
        #  the new window size will be.
        if window_size is None:
            window_size = self.__proxy__['window_size']

        else:
            if not isinstance(window_size, int):
                raise TypeError("Input 'window_size' must be an integer.")

            if window_size < 1:
                raise ValueError("Input 'window_size' must greater than or " +
                                 "equal to 1.")

        ## Validate and determine the `min_observations` parameter.
        if min_observations is None:
            min_observations = self.__proxy__['min_observations']

        else:
            if not isinstance(min_observations, int):
                raise TypeError("If specified, input 'min_observations' must " +
                                "be a positive integer.")

            if min_observations < 1:
                raise ValueError("If specified, input 'min_observations' must " +
                                 "be a positive integer.")


        ## TimeSeries-specific dataset validation
            ## Make the sure new data occurs *after* the existing data.
        scores = self.__proxy__['scores']

        if isinstance(dataset, _gl.TimeSeries):
            first_new_timestamp = dataset[0][dataset.index_col_name]
            last_old_timestamp = scores[-1][scores.index_col_name]

            if first_new_timestamp < last_old_timestamp:
                raise _ToolkitError("The new dataset has data with " +
                                    "earlier timestamps than the existing " +
                                    "dataset. Please ensure that new data " +
                                    "occurs after existing data.")


        ## Extract the feature from the new dataset and validate it.
        feature = self.__proxy__['feature']

        try:
            series = dataset[feature]
        except:
            raise _ToolkitError("The feature specified by the original " +
                                "model could not be found in the input " +
                                "'dataset'.")

        if not series.dtype() in [int, float]:
            raise ValueError("The values in the specified feature must be " +
                             "integers or floats.")


        ## Create a new model and cut the old score object to the window size.
        new_state = {k: self.__proxy__[k]
            for k in ['verbose', 'feature', 'dataset_type']}

        new_state['window_size'] = window_size
        new_state['min_observations'] = min_observations

        new_model = MovingZScoreModel(new_state)


        ## Save just the old data needed for the moving statistics on the new
        #  data.
        if len(scores) < window_size:
            old_scores = scores[:]
        else:
            old_scores = scores[-window_size:]


        ## Compute Z-scores and anomaly scores.
        series = old_scores[feature].append(series)
        moving_average, moving_zscore, sufficient_data = \
            _moving_z_score(series, window_size, min_observations)

        anomaly_score = abs(moving_zscore)

        if not sufficient_data:
            logger.warning("The number of observations is smaller than " +
                           "the minimum number needed to compute a " +
                           "moving Z-score, so all anomaly scores are 'None'. " +
                           "Consider adding more data with the model's `update` " +
                           "method, or reducing the `window_size` or " +
                           "`min_observations` parameters.")

        ## General post-processing and formatting.
        scores = _gl.SFrame({feature: series,
                             'moving_average': moving_average,
                             'anomaly_score': anomaly_score})
        scores['model_update_time'] = _dt.datetime.now()

        scores = scores[[feature,  # reorder the columns
                         'moving_average',
                         'anomaly_score',
                         'model_update_time']]


        ## Replace the new Z-scores for the *old* data with the original
        #  Z-score for that data.
        num_new_examples = len(dataset)
        new_scores = scores[-num_new_examples:]

        if isinstance(dataset, _gl.TimeSeries):
            new_scores[dataset.index_col_name] = dataset[dataset.index_col_name]
            new_scores = _gl.TimeSeries(new_scores, index=dataset.index_col_name)

            ## The index column should have the same name in the old and new
            #  data. If it doesn't, change the name in the old scores.
            if dataset.index_col_name != old_scores.index_col_name:
                old_scores = old_scores.rename(
                           {old_scores.index_col_name: dataset.index_col_name})

                if verbose:
                    logger.warning("The new dataset's index column name " +
                                   "does not match the existing index " +
                                   "column name. The new name is used in " +
                                   "the new model.")

            final_scores = old_scores.union(new_scores)

        else:
            new_scores = new_scores.add_row_number('row_id')
            old_scores['row_id'] = None
            old_scores['row_id'] = old_scores['row_id'].astype(int)
            final_scores = old_scores.append(new_scores)


        ## Finalize and return the model.
        new_model.__proxy__['num_examples'] = len(scores)
        new_model.__proxy__['scores'] = final_scores
        new_model.__proxy__['training_time'] = _time.time() - start_time

        return new_model

    def _get_version(self):
        return self._PYTHON_MOVING_ZSCORE_VERSION

    def _save_impl(self, pickler):
        """
        Save the model as a directory, which can be loaded with the
        :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        pickler : GLPickler
            An opened GLPickle archive (Do not close the archive).

        See Also
        --------
        graphlab.load_model

        Examples
        --------
        >>> model.save('my_model_file')
        >>> loaded_model = graphlab.load_model('my_model_file')
        """

        ## The GL pickler does not support TimeSeries, so we need to convert
        #  and un-convert to SFrame here. Furthermore, the proxy does not
        #  support copying, so we need to change proxy itself, then change it
        #  back.
        if self.__proxy__['dataset_type'] == 'TimeSeries':
            self.__proxy__['index_col_name'] = self.__proxy__['scores'].index_col_name
            self.__proxy__['scores'] = self.__proxy__['scores'].to_sframe()

            pickler.dump(self.__proxy__)

            self.__proxy__['scores'] = _gl.TimeSeries(self.__proxy__['scores'],
                                        index=self.__proxy__['index_col_name'])
            self.__proxy__.pop('index_col_name')

        else:
            pickler.dump(self.__proxy__)

    @classmethod
    def _load_version(self, unpickler, version):
        """
        A function to load a previously saved MovingZScoreModel
        instance.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handler.

        version : int
            Version number maintained by the class writer.
        """
        state = unpickler.load()

        if state['dataset_type'] == 'TimeSeries':
            state['scores'] = _gl.TimeSeries(state['scores'],
                                             index=state['index_col_name'])
            state.pop('index_col_name')

        if version == 0:
            state['min_observations'] = None

        return MovingZScoreModel(state)

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the MovingZScoreModel.
        """
        return self.__repr__()

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """

        width = 40
        key_str = "{:<{}}: {}"

        sections, section_titles = self._get_summary_struct()
        accessible_fields = {
            "scores": "Anomaly score for each instance in the current dataset."}

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
            ('Dataset type', 'dataset_type'),
            ('Feature', 'feature'),
            ('Number of examples', 'num_examples'),
            ('Moving window size', 'window_size'),
            ('Minimum required observations', 'min_observations')]

        training_fields = [
            ('Total training time (seconds)', 'training_time')]

        section_titles = ['Schema', 'Training summary']
        return([model_fields, training_fields], section_titles)

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current MovingZScoreModel instance.
        """
        return {k: self.__proxy__[k] for k in get_default_options()['name']}

    def show(self):
        """
        Visualize the model scores. This is particularly useful for visualizing
        the distribution of anomaly scores.
        """

        if self.__proxy__['dataset_type'] == 'TimeSeries':
            self.__proxy__['scores'].to_sframe().show()
        else:
            self.__proxy__['scores'].show()
