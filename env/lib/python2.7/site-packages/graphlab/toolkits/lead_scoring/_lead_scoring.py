"""
Methods for creating and working with a lead scoring model.
"""

import logging as _logging
logger = _logging.getLogger('graphlab.toolkits.lead_scoring')

import copy as _copy
import math as _math
import time as _time
import datetime as _dt
from collections import OrderedDict as _OrderedDict
import array as _array

import graphlab as _gl
import graphlab.aggregate as _agg
import graphlab.connect as _mt

from graphlab.toolkits._model import CustomModel as _CustomModel
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy

from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _raise_error_if_sframe_empty
from graphlab.toolkits._private_utils import _summarize_accessible_fields
from graphlab.toolkits._private_utils import _robust_column_name



## HELPER FUNCTIONS
## ----------------------------------------------------------------------------
def _rename_dataset_by_schema(dataset, schema):
    """
    Rename an SFrame or TimeSeries according to a user-provided schema, without
    changing the user's original input dataset.

    Parameters
    ----------
    dataset : SFrame or TimeSeries
        Input data.

    schema : dict{str: str}
        Input schema. Keys are "standardized" field names, values are the
        actual names in the input dataset. *Assumes* only strings as values,
        i.e. remember to take out the 'features' entry from a schema before
        passing it here.

    Returns
    -------
    _dataset : SFrame
        Renamed SFrame.
    """
    _dataset = dataset[schema.values()]
    _dataset = _dataset.rename({v: k for k, v in schema.items()})
    return _dataset

def _construct_dataset_schema(data, valid_keys, required_keys, name):
    """
    When a user's input dataset contains the required columns, the accompanying
    schema is optional. This function constructs a simple "identity" schema to
    make the rest of the `create` code simpler.

    NOTE: the schema produced by this function does not include a 'features'
    entry - that is processed (possibly created) later in the `create`
    function.

    Parameters
    ----------
    data : SFrame
        Input data.

    valid_keys : dict(str: tuple(type))
        Valid keys for the schema.

    required_keys : set
        Keys that must be present in the schema.

    name : str
        Name of the dataset under inspection, for better error messages.

    Returns
    -------
    schema : dict
        The implicit account schema.
    """

    ## Add every valid schema key that is in the dataset, *except* 'features',
    #  to the constructed schema.
    schema = {}

    for key, allowed_types in valid_keys.items():
        if key != 'features':
            if (key in data.column_names() and
                data[key].dtype() in allowed_types):

                schema[key] = key

    ## Check that the required keys are present and have the right types.
    if not required_keys.issubset(schema.keys()):
        msg = ("The '{}' dataset does not have the required columns. " +
               "Please explicitly specify the associated schema.").format(name)
        raise _ToolkitError(msg)


    ## Create the features entry from the columns that are left.
    schema['features'] = _identify_valid_features(data,
                               LeadScoringModel.valid_account_keys['features'],
                               exclude_columns=schema.values())

    return schema

def _validate_dataset_schema(data, schema, valid_keys, required_keys, name):
    """
    Input validation for a dataset against a user-provided schema.

    Parameters
    ----------
    data : SFrame
        Input dataset.

    schema : dict
        Input dataset schema.

    valid_types : list[type]
        Allowed types for the data object.

    valid_keys : dict(str: tuple(type))
        Valid keys for the schema, mapped to the valid types for the
        corresponding column in the dataset.

    required_keys : set
        Keys that must be present in the schema.

    name : str
        Name of the input dataset, for more helpful messages on any errors that
        must be raised.
    """

    if not isinstance(schema, dict):
        raise TypeError("The '{}' schema must be a dictionary.".format(name))

    ## Check that all keys are valid.
    if not set(schema.keys()).issubset(valid_keys.keys()):
        invalid_keys = set(schema.keys()).difference(valid_keys.keys())
        message_template = "The '{0}' schema has invalid keys: {1}"
        message = message_template.format(name, ', '.join(invalid_keys))
        raise _ToolkitError(message)

    ## Are all required keys present?
    if not required_keys.issubset(schema.keys()):
        missing_keys = required_keys.difference(schema.keys())
        message_template = "The '{0}' schema is missing required keys: {1}"
        message = message_template.format(name, ', '.join(missing_keys))
        raise _ToolkitError(message)

    ## Check that schema values are actually columns in the input dataset and
    #  that those columns have allowed types.
    for key, val in schema.items():

        if not isinstance(val, (str, list)):
            msg = ("The '{0}' schema has an invalid value '{1}'. " +
                   "All schema values must be strings or lists of strings.")
            raise TypeError(msg.format(name, val))

        if isinstance(val, str):
            val = [val]

        for col in val:
            if not col in data.column_names():
                msg = ("For the '{}' schema, column '{}' does not exist in " +
                       " the provided dataset.")
                raise _ToolkitError(msg.format(name, col))

            if data[col].dtype() not in valid_keys[key]:
                msg = ("For the '{}' schema, column '{}' has an invalid type " +
                       "in the corresponding dataset. Allowed types are: {}.")
                raise _ToolkitError(msg.format(name, col, valid_keys[key]))

def _validate_conversion_status(data):
    """
    Type checking for the target variable in the accounts table.

    Parameters
    ----------
    data : SArray
        Data in the conversion status column.
    """
    if data.dtype() != int:
        raise _ToolkitError("The 'conversion_status' column of the 'accounts' " +
                            "SFrame must contain integers.")

    if data.min() < -1 or data.max() > 1:
        msg = ("Values in the 'conversion_status' column must be " +
               "-1 (failure), 1 (successful conversion), or 0 (open account).")
        raise _ToolkitError(msg)

def _identify_valid_features(data, valid_types, exclude_columns):
    """
    Extract the column names in the 'data' dataset that are valid features,
    based on allowed types and a blacklist of column names.

    Parameters
    ----------
    data : SFrame or TimeSeries
        Input dataset.

    valid_types : list[type]
        Allowed types for features. Columns that have other features are
        ignored.

    exclude_columns : list[str]
        Columns to black-list from the output feature list (typically because
        they're already being used, e.g. as the target column).

    Returns
    -------
    feature_names : list[str]
        List of column names in 'data' that have the appropriate type and
        aren't a priori blacklisted.
    """
    if isinstance(data, _gl.SFrame):
        col_names = data.column_names()
    elif isinstance(data, _gl.TimeSeries):
        col_names = data.value_col_names
    else:
        msg = ("Auto-feature identification only works for SFrames and " +
               "TimeSeries.")
        raise TypeError(msg)

    return [c for c in col_names if data[c].dtype() in valid_types and c not in exclude_columns]

def _describe_feature_path(path):
    """
    Convert the decision tree navigation object's symbolic description of a
    split into an English description.

    Parameters
    ----------
    path : str
        String of symbols describing a split in a decision tree.

    Returns
    -------
    out
        An English version of the split logic.
    """
    description = []

    for v in path:
        if v['node_type'] == 'indicator':
            if v['index'] == '':
                description.append("{} {} ''".format(v['feature'], v['sign']))
            else:
                description.append("{} {} {}".format(v['feature'], v['sign'],
                                                     v['index']))
        else:
            description.append("{} {} {}".format(v['feature'], v['sign'],
                                                 v['value']))

    return description

def _assign_accounts_to_segments(tree, accounts):
    """
    Assign a set of accounts to segments based on a decision tree.

    Parameters
    ----------
    tree : graphlab.toolkits.decision_tree_regression.DecisionTreeRegression
        The segmentation model.

    accounts : SFrame
        The accounts to segment. Must have a column of fitted or predicted
        conversion probabilities, called 'conversion_prob'.

    Returns
    -------
    segment_assignments : SFrame
        Each account in `accounts`, but now with a segment ID. Only the
        metadata_cols plus the segment ID are retained from the original
        `accounts` table.
    """
    fitted_node_arrays = tree.extract_features(accounts)
    segment_assignments = _gl.SArray([x[0] for x in fitted_node_arrays]).astype(int)
    return segment_assignments

def _compute_segment_stats(tree, segment_assignments):
    """
    Compute summary statistics about the training account segments.

    Parameters
    ----------
    tree : graphlab.toolkits.decision_tree_regression.DecisionTreeRegression
        Segmentation model.

    segment_assignments : SFrame
        Account information, primarily which segment each account belongs to
        ('segment_id') and the fitted or predicted conversion probability
        ('conversion_prob') for the account.

    Returns
    -------
    segment_stats : SFrame
        One row per segment, with descriptive stats.
    """
    tree_nav = tree._get_tree()

    leaves = [k for k, v in tree_nav.nodes.items() if v.is_leaf]
    leaf_values = [tree_nav.nodes[k].value + 0.5 for k in leaves]
    leaf_paths = [tree_nav.get_prediction_path(k) for k in leaves]
    leaf_path_descriptions = [_describe_feature_path(p) for p in leaf_paths]

    segment_aggregator = {
        'num_training_accounts': _agg.COUNT('segment_id'),
        'mean_conversion_prob': _agg.MEAN('conversion_prob'),
        'stdev_conversion_prob': _agg.STD('conversion_prob'),
        'median_conversion_prob': _agg.QUANTILE('conversion_prob', 0.5),
        'min_conversion_prob': _agg.MIN('conversion_prob'),
        'max_conversion_prob': _agg.MAX('conversion_prob')}

    segment_stats = segment_assignments.groupby('segment_id', segment_aggregator)
    segment_stats['median_conversion_prob'] = segment_stats['median_conversion_prob'].apply(lambda x: x[0])

    segment_features = _gl.SFrame({'segment_id': leaves,
                                   'segment_features': leaf_path_descriptions})
    segment_stats = segment_stats.join(segment_features, on='segment_id', how='inner')
    segment_stats = segment_stats.sort('mean_conversion_prob', ascending=False)

    return segment_stats

def _find_implicit_failure_accounts(training_accounts, open_accounts,
                                    trial_duration, quantile=0.95):
    """
    Determine which of the open accounts have implicitly failed because they've
    been open for too long. "Too long" is based on the distribution of time-to-
    conversion of the training accounts, both successes and failures, as well
    as the trial period duration, if indicated by end user. An open cannot that
    is still in its trial period will not be declared to be an implicit
    failure, no matter what the distribution of time-to-conversion is for the
    training accounts.

    Parameters
    ----------
    training_accounts : SFrame
        Training accounts, that have a decided conversion status.

    open_accounts : SFrame
        Accounts that don't have a decided conversion status yet (the ones we
        want to predict).

    trial_duration : datetime.timdelta
        Length of the product trial. An open account that's still in its trial
        period will not be considered an implicit failure, regardless of the
        distribution of time-to-conversion in the training accounts.

    quantile : float
        The quantile of the training days-to-conversion to use as the threshold
        for determining implicit failure of an open account.

    Returns
    -------
    implicit_fail_mask : _gl.SArray
        A boolean mask, indicating which of the open accounts have implicitly
        failed.
    """

    ## Compute the duration for implicit failure. NOTE: 'time_to_conversion' is
    #  not the same as 'open_duration', which is bounded by the trial duration,
    #  if that's provided.
    time_to_conversion = (training_accounts['decision_date'] -
                          training_accounts['open_date'])
    implicit_fail_threshold = (time_to_conversion.sketch_summary()
                                                 .quantile(quantile))

    ## If trial duration is provided, it's the lower bound on implicit failure
    #  - an account that's still in the trial period cannot be called an
    #  implicit failure.
    if trial_duration is not None:
        implicit_fail_threshold = max(implicit_fail_threshold,
                                      trial_duration.total_seconds())

    # Filter the accounts that have implicitly failed.
    now = _dt.datetime.now()
    open_duration = now - open_accounts['open_date']
    implicit_fail_mask = open_duration >= implicit_fail_threshold

    return implicit_fail_mask

def _compute_open_duration(x, trial_duration):
    """
    Lambda function for computing the number of days a given account (`x`) was
    open during the trial period. Note that taking the min of two
    datetime.timedelta's automatically converts the answer to seconds, at least
    in the form of an SFrame apply.

    The value returned by this function is used to determine invalid training
    accounts and to standardize the interaction features to daily averages.
    """
    if x['conversion_status'] == 0 or x['decision_date'] is None:
        open_duration = _dt.datetime.now() - x['open_date'] + _dt.timedelta(days=1)
    else:
        open_duration = x['decision_date'] - x['open_date'] + _dt.timedelta(days=1)

    if trial_duration is not None:
        open_duration = min(open_duration, trial_duration)

    return open_duration.total_seconds() / (24 * 60 * 60)

def _max_leaves_to_max_depth(max_leaves):
    """
    Convert the an upper bound on the number of leaves in a tree to the maximum
    depth of the tree.

    Parameters
    ----------
    max_leaves : int
        Maximum number of leaves allowed in a binary tree.

    Returns
    -------
    int
        Maximum depth of the binary tree such that there are not more than
        `max_leaves` leaves.
    """
    return int(_math.floor(_math.log(max_leaves, 2)))



## TOOLKIT FUNCTIONS
## ----------------------------------------------------------------------------

def get_default_options():
    """
    Default values for the create function.
    """
    params = [
        {'name': 'trial_duration',
         'default_value': '',
         'description': 'Duration of the evaluation/prediction period.',
         'lower_bound': '',
         'upper_bound': '',
         'parameter_type': 'datetime.timedelta'},

        {'name': 'account_schema',
         'default_value': '',
         'description': 'Schema for the "accounts" SFrame.',
         'lower_bound': '',
         'upper_bound': '',
         'parameter_type': 'dict'},

        {'name': 'interaction_schema',
         'default_value': '',
         'description': 'Schema for the "interactions" table.',
         'lower_bound': '',
         'upper_bound': '',
         'parameter_type': 'dict'},

        {'name': 'max_segments',
         'default_value': 8,
         'description': 'Maximum number of market segments.',
         'lower_bound': 1,
         'upper_bound': '',
         'parameter_type': 'int'},

        {'name': 'verbose',
         'default_value': 'True',
         'description': 'Verbose printing flag.',
         'lower_bound': 'False',
         'upper_bound': 'True',
         'parameter_type': 'bool'}]

    return _gl.SFrame(params).unpack('X1', column_name_prefix='')

def create(accounts, account_schema=None,
           interactions=None, interaction_schema=None,
           trial_duration=None, max_segments=8, verbose=True, **kwargs):
    """
    Create a 'LeadScoringModel', which does two main things:

        1. score open accounts in terms of their probability of a successful
           conversion, using a gradient boosted trees classifier.

        2. segment accounts into similar groups based on the relationship
           between account features, usage, and conversion status. The
           segmentation is created from a decision tree model trained on the
           fitted values of the training accounts in the scoring model.

    Our philosophy is that every account is *open*---i.e. undecided---unless
    specified otherwise. *Successful* conversions must be explicitly indicated
    by the user, while *failures* may be indicated explicitly or, if
    interaction data is provided, the model may automatically determine
    accounts that have failed implicitly.

    The goal of the tool is to accurately predict the conversion status for
    open accounts, based on the patterns among the existing successful and
    failed accounts. At a minimum, these patterns are based on metadata about
    each account (provided in the 'accounts' table), however the model also
    accepts information about interactions between accounts and your business
    assets, such as product, websites, and email campaigns.

    If this "interaction" data is provided, the lead scoring tool automatically
    creates relevant features to improve scoring and segmentation accuracy.
    In this case, a 'trial_duration' parameter must be specified, along with an
    'open_date' for each account. The model will only use the interactions
    occurring during the specified trial period to create the usage-based
    features. If 'interactions' are specified, each account must also have a
    'decision_date'; the model uses this information to determine if open
    accounts have been open for so long that they can be treated as implicit
    examples of failed accounts, improving the model's power.

    .. warning:: The lead scoring toolkit is in beta, and may change in future
        versions of GraphLab Create. Feedback is welcome: please send comments
        to product-feedback@turi.com.

    Parameters
    ----------
    accounts : SFrame
        Data about sales accounts. Must include the columns specified by the
        'account_schema' argument, described below.

    account_schema : dict, optional
        Specifies columns in the 'accounts' SFrame to be used by the model.
        This dictionary allows only a small, fixed set of keys, each of which
        must be mapped to a column or columns in the 'accounts' table. In the
        following table, 'Type' is the data type of the corresponding column in
        the 'accounts' SFrame. Please see the Examples section as well.

        ================= ======== ============================================
        Key               Type     Description
        ================= ======== ============================================
        account_id        str|int  Unique ID for each account. Required.
        conversion_status int      Whether the account has converted
                                   successfully (1), failed (-1) or remains
                                   open (0). Required.
        decision_date     datetime Date on which the account was determined to
                                   convert (or not). Required if an
                                   'interactions' dataset is provided. Values
                                   for the open accounts are ignored.
        open_date         datetime Date on which the account opened. Required
                                   if an 'interactions' dataset is provided.
        account_name      str      Human-readable name for an account. This
                                   makes it easier to understand the model's
                                   output.
        features                   Additional account metadata columns to use
                                   as features. If not specified, all columns
                                   not otherwise accounted for are considered
                                   for use by the tool. Features may be numeric
                                   (float or int), categorical (str), arrays of
                                   floats, or dictionaries mapping strings to
                                   integers or floats.
        ================= ======== ============================================

        NOTE: The 'accounts' table must contain at least some successful and
        accounts, in addition to any open accounts. If 'interactions' data is
        provided account failures can be determined implicitly, otherwise the
        'accounts' table must also include some account failures.

        NOTE: if the 'accounts' table contains columns with the names used in
        the 'account_schema', the schema does not have to be specified. The
        appropriate columns will be used automatically.

    interactions : TimeSeries, optional
        Information about interactions between accounts and aspects of your
        business, typically individual products, web pages, or marketing
        materials. Each row represents a single interaction between an account
        and a business item, at a specific point in time. The model uses this
        information to construct usage-based features for each account:

            - *num_events*: number of interactions per day

            - *num_users*: number of unique users per day (if 'user' is
              specified)

            - *num_items*: number of unique items per day (if 'item' is
              specified).

        All features are computed only for interactions occurring during the
        specified trial period. For training accounts, this is defined to start
        at the account's 'open_date' and end at the 'open_date' +
        'trial_duration'. For open accounts, the features are based on
        interactions occurring between the open date and today's date.

        If provided, the 'interactions' TimeSeries must include the columns
        specified by the 'interaction_schema' parameter, described below.

    interaction_schema : dict, optional
        Schema for the 'interactions' TimeSeries. Required if the
        'interactions' parameter is specified, and ignored if 'interactions' is
        not provided. The following fields may be defined in this dictionary;
        note that *either* 'user' or 'account_id' must be specified, to connect
        the interactions to the account information.

        ================= ======== ============================================
        Key               Type     Description
        ================= ======== ============================================
        account_id        str|int  Account ID. Required.
        user              str|int  User ID.
        item              str|int  Item ID.
        features                   Additional interaction feature columns. If
                                   not specified, all columns not otherwise
                                   accounted for are considered for use by the
                                   tool. Features may be numeric (float or
                                   int), categorical (str), arrays of floats,
                                   or dictionaries mapping strings to integers
                                   or floats.
        ================= ======== ============================================

        NOTE 1: if the 'interactions' table contains columns with the names
        used in the 'interaction_schema', this schema does not have to be
        specified. The appropriate columns will be used automatically.

    trial_duration : datetime.timedelta, optional
        Length of the sales trial period. If specified, only interactions that
        occur during the trial period (from the 'open_date' to that date plus
        the trial duration) are used to create usage features. This parameter
        is required if the 'interactions' table is provided.

    max_segments : int
        Maximum number of segments to create with the tool's segmentation
        model. Because the segmentation model is a binary tree, the actual
        number of segments will likely differ slightly from the argument to
        this parameter.

    verbose : bool, optional
        If True, print progress updates and model details during model
        creation.

    **kwargs : optional
        Additional keyword arguments are passed to the creation of the internal
        scoring model, which is a boosted trees classifier.

    Returns
    -------
    out : LeadScoringModel
        The primary outputs of the LeadScoringModel are the
        'open_account_scores', 'training_account_scores', and
        'segment_descriptions' attributes.

        - *open_account_scores*: predicted conversion probability for each open
          account and market segment assignment.

        - *training_account_scores*: estimated conversion probability (i.e.
          fitted value) and market segment assignment for each "training"
          account, i.e. each account that has either converted or failed. This
          includes both explicit (user-specified) and implicit
          (model-determined) failures.

        - *segment_descriptions*: statistics about each market segment, based
          on the model's estimated conversion probabilities for the training
          accounts.

    See Also
    --------
    LeadScoringModel,
    graphlab.toolkits.boosted_trees_classifier
    graphlab.toolkits.decision_tree_regression
    toolkits.churn_prediction,

    References
    ----------
    - Turi User Guide. `Lead Scoring <https://turi.com/learn/userguide/lead_scoring/lead_scoring.html>`_

    Notes
    -----
    - The segment IDs in the toolkit output SFrames do not typically start at
      1, because they correspond to the leaves of the decision tree in the
      internal segmentation model.

    Examples
    --------
    >>> accounts = graphlab.SFrame({'name': ['Acme', 'Duff', 'Oscorp', 'Tyrell'],
    ...                             'id': [0, 1, 2, 3],
    ...                             'status': [1, -1, 1, 0],
    ...                             'num_employees': [100, 250, 3000, 1132],
    ...                             'product_usage': [1327, 554, 87121, 12755]})
    ...
    >>> account_schema = {
    ...     'conversion_status': 'status',
    ...     'account_name': 'name',
    ...     'account_id': 'id',
    ...     'features': ['num_employees', 'product_usage']}
    ...
    >>> model = graphlab.lead_scoring.create(accounts, account_schema)
    ...
    >>> print model.open_account_scores
    +--------------+------------+---------------+---------------+--------------------+
    | account_name | account_id | num_employees | product_usage |  conversion_prob   |
    +--------------+------------+---------------+---------------+--------------------+
    |    Tyrell    |     3      |      1132     |     12755     | 0.8005750775337219 |
    +--------------+------------+---------------+---------------+--------------------+
    +------------+
    | segment_id |
    +------------+
    |     2      |
    +------------+
    """
    start_time = _time.time()
    _mt._get_metric_tracker().track('toolkit.lead_scoring.create')

    logger.warning("The lead scoring toolkit is in beta, and may change in " +
        "future versions of GraphLab Create. Feedback is welcome: please send " +
        "comments to product-feedback@turi.com.")


    #######################################
    ## ACCOUNT AND GENERIC PREPROCESSING ##
    #######################################

    ## Validate inputs.
    ## ----------------
    ## Validate the input data types.
    if not isinstance(accounts, _gl.SFrame):
        raise TypeError("The 'accounts' dataset must be an SFrame.")

    _raise_error_if_sframe_empty(accounts, "accounts")

    ## trial_duration must be a datetime.timedelta >= 1 day.
    if trial_duration is not None:
        if not isinstance(trial_duration, _dt.timedelta):
            msg = "The 'trial_duration' parameter must be a 'datetime.timedelta'."
            raise TypeError(msg)

        if trial_duration < _dt.timedelta(1):
            msg = ("The 'trial_duration' parameter must be a at least one " +
                   "day.")
            raise ValueError(msg)

    ## max_segments must be an integer >= 1.
    if not isinstance(max_segments, int):
        raise TypeError("The 'max_segments' parameter must be an integer.")

    if max_segments < 1:
        raise ValueError("The 'max_segments' parameter must be a least 1.")

    ## Make the account schema if not provided, otherwise validate the schema
    #  against the provided dataset.
    if account_schema is None:
        account_schema = _construct_dataset_schema(accounts,
                                        LeadScoringModel.valid_account_keys,
                                        LeadScoringModel.required_account_keys,
                                        name='accounts')

    else:
        _validate_dataset_schema(accounts, account_schema,
                                LeadScoringModel.valid_account_keys,
                                LeadScoringModel.required_account_keys,
                                name='accounts')

        if 'features' not in account_schema.keys():
            account_schema['features'] = _identify_valid_features(accounts,
                               LeadScoringModel.valid_account_keys['features'],
                               exclude_columns=account_schema.values())


    ## Make sure there are no duplicate account IDs.
    if len(accounts[account_schema['account_id']].unique()) != len(accounts):
        msg = ("Duplicate account IDs detected. Each row of the " +
               "'accounts' table should correspond to a unique account " +
               "with a unique account ID string.")
        raise _ToolkitError(msg)


    ## Rename columns to make the code easier to read and so the output scores
    #  tables are consistent with the output schemas.
    temp_account_schema = {k: v for k, v in account_schema.items()
                           if k != 'features'}
    _accounts = _rename_dataset_by_schema(accounts, temp_account_schema)

    for c in account_schema['features']:
        _accounts[c] = accounts[c]


    ## Validate the conversion status.
    _validate_conversion_status(_accounts['conversion_status'])


    ## If 'open_date' and 'decision_date' are provided (as they must be, if
    #  interaction data is provided), make sure the conversion date is after
    #  the open date.
    invalid_accounts = []
    open_duration_name = _robust_column_name('open_duration',
                                             _accounts.column_names())

    if ('open_date' in account_schema.keys() and
        'decision_date' in account_schema.keys()):

        msg = ("Checking for invalid accounts (conversion date before " +
               "open date)....")
        if verbose: logger.info(msg)

        _accounts[open_duration_name] = _accounts.apply(
                           lambda x: _compute_open_duration(x, trial_duration))
        invalid_date_mask = _accounts[open_duration_name] <= 0

        if any(invalid_date_mask):
            msg = ("Found {} invalid accounts, with decision dates " +
                   "*before* open dates. These accounts are being removed " +
                   "from the training set; their IDs (or row numbers) are " +
                   "available in the model's 'invalid_accounts' field.")
            if verbose: logger.warning(msg.format(sum(invalid_date_mask)))

            invalid_accounts = _accounts['account_id'][invalid_date_mask]

            valid_date_mask = 1 - invalid_date_mask
            valid_date_mask = valid_date_mask.fillna(1)
            _accounts = _accounts[valid_date_mask]



    ########################################
    ## INTERACTION-SPECIFIC PREPROCESSING ##
    ########################################
    close_date_name = _robust_column_name('close_date', _accounts.column_names())
    gbt_features = account_schema['features'][:]
    num_interactions = 0
    num_interaction_features = 0

    if interactions is not None:

        ## Validate inputs
        ## ---------------
        num_interactions = len(interactions)

        ## Validate input data types
        if not isinstance(interactions, _gl.TimeSeries):
            raise TypeError("The 'interactions' dataset must be a TimeSeries.")

        _raise_error_if_sframe_empty(interactions._sframe, "interactions")


        ## Make the interaction schema if not provided, otherwise validate the
        #  schema against the provided dataset.
        if interaction_schema is None:
            interaction_schema = _construct_dataset_schema(interactions,
                                    LeadScoringModel.valid_interaction_keys,
                                    LeadScoringModel.required_interaction_keys,
                                    name='interactions')

        else:
            _validate_dataset_schema(interactions, interaction_schema,
                                    LeadScoringModel.valid_interaction_keys,
                                    LeadScoringModel.required_interaction_keys,
                                    name='interactions')

            if not 'features' in interaction_schema.keys():
                interaction_schema['features'] = _identify_valid_features(
                       interactions,
                       LeadScoringModel.valid_interaction_keys['features'],
                       exclude_columns=interaction_schema.values())

        num_interaction_features = len(interaction_schema['features'])


        ## If interactions are provided open date, and conversion date are
        #  specified in the accounts table, and the trial_duration is
        #  specified.
        if not 'open_date' in account_schema.keys():
            msg = ("If the 'interactions' table is provided, the account open" +
                   "date ('open_date') must be specified in the account schema.")
            raise _ToolkitError(msg)

        if not 'decision_date' in account_schema.keys():
            msg = ("If the 'interactions' table is provided, the account " +
                   " decision date must be specified in the account schema. " +
                   "For open accounts, the values in this column are ignored.")
            raise _ToolkitError(msg)

        if trial_duration is None:
            msg = ("If 'interactions' data are provided, a 'trial_duration' " +
                   "must be specified. In tandem with the account 'open_date' " +
                   "values, this is used to define a 'trial period'; only " +
                   "interactions fall in this period are used to construct " +
                   "features.")
            raise _ToolkitError(msg)


        ## Rename columns to make the code easier to read.
        temp_interaction_schema = {k: v for k, v in interaction_schema.items()
                                   if k != 'features'}
        _interactions = _rename_dataset_by_schema(interactions,
                                                  temp_interaction_schema)

        for c in interaction_schema['features']:
            _interactions[c] = interactions[c]


        ## Engineer interaction features
        ## -----------------------------
        if verbose: logger.info("Constructing interaction-based features....")

        ## Define the the close date.
        _accounts[close_date_name] = _accounts['open_date'] + trial_duration

        ## Join the open and close dates to the interactions data. NOTE: this
        #  could be faster for big sets of interactions if I aggregate feature
        #  counts by day first. NOTE 2: think about unpacking items and users
        #  here.
        _interactions = _interactions.join(_accounts, on='account_id',
                                           how='inner')

        ## Remove interactions that happen outside of the trial period.
        timestamp = interactions.index_col_name

        end_mask = _interactions[timestamp] <= _interactions[close_date_name]
        start_mask = _interactions[timestamp] >= _interactions['open_date']
        mask = start_mask * end_mask

        trial_interactions = _interactions[mask]

        ## Define and compute the interaction-based features.
        ## Possible other features:
        # - number of days used/days open.
        ftr = _robust_column_name('num_events', _accounts.column_names())
        feature_aggregator = {ftr: _agg.COUNT(timestamp)}

        if 'user' in interaction_schema.keys():
            ftr = _robust_column_name('num_users', _accounts.column_names())
            feature_aggregator[ftr] = _agg.COUNT_DISTINCT('user')

        if 'item' in interaction_schema.keys():
            ftr = _robust_column_name('num_items', _accounts.column_names())
            feature_aggregator[ftr] = _agg.COUNT_DISTINCT('item')

        interaction_stats = trial_interactions.groupby('account_id',
                                                       feature_aggregator)

        ## Join the features back to the accounts data (all of it, not just
        #  training accounts).
        _accounts = _accounts.join(interaction_stats, on='account_id',
                                   how='left')

        ## Fill in zeros for counts with no interactions and divide by the
        #  number of days the account was open during the trial period.
        # - NOTE: remember 'open_duration' is present as long as
        #   'decision_date' and 'open_date' are specified for the accounts,
        #   which they must be if 'interactions' data is provided.
        for c in feature_aggregator.keys():
            try:
                _accounts[c] = _accounts[c].fillna(0)
                _accounts[c] = _accounts[c] / _accounts[open_duration_name]
            except:
                pass

        ## Update the feature set.
        gbt_features = gbt_features + list(feature_aggregator.keys())



    ########################
    ## MODEL CONSTRUCTION ##
    ########################

    ## Decide if any of the open accounts are actually implicit failures.
    ## ------------------------------------------------------------------
    ## Separate out the open accounts.
    training_accounts = _accounts.filter_by([1, -1], 'conversion_status')
    open_accounts = _accounts.filter_by(0, 'conversion_status')

    num_open_accounts = len(open_accounts)
    num_training_accounts = len(training_accounts)

    num_successes = sum(training_accounts['conversion_status'] == 1)
    num_explicit_failures = len(training_accounts) - num_successes
    num_implicit_failures = 0
    num_total_failures = num_explicit_failures

    ## Need the opening and conversion dates to determine implicit failures.
    #  This should always be the case if there are interactions. It may or may
    #  not be the case if only account data is provided.
    if ('open_date' in account_schema.keys() and
        'decision_date' in account_schema.keys()):

        msg = "Looking for accounts that have failed implicitly...."
        if verbose: logger.info(msg)

        implicit_fail_mask = _find_implicit_failure_accounts(training_accounts,
                                                             open_accounts,
                                                             trial_duration, # this could be None
                                                             quantile=0.95)

        ## Append implicit failures to the training data (but leave them in the
        #  open accounts as well, so they can still be scored by the model).
        implicit_fail_accounts = open_accounts[implicit_fail_mask]
        implicit_fail_accounts['conversion_status'] = -1
        training_accounts = training_accounts.append(implicit_fail_accounts)

        num_implicit_failures = len(implicit_fail_accounts)
        num_total_failures = num_implicit_failures + num_explicit_failures

        if num_implicit_failures > 0:
            msg = ("Found {} open accounts that have failed implicitly. " +
                   "These accounts have been open longer than the time most " +
                   "training accounts take to reach a decision. " +
                   "These implicit failures will be added to the training " +
                   "dataset, but will still be scored as open accounts.")
            if verbose: logger.warning(msg.format(num_implicit_failures))

    ## Check that there are at least some training successes and failures.
    if num_successes == 0:
        msg = ("The final account data has no successes to " +
               "learn from. Please specify which accounts have successfully " +
               "converted.")
        raise _ToolkitError(msg)

    if num_total_failures == 0:
        msg = ("The final account data has no failures to learn from. Please " +
               "explicitly specify accounts that have failed to convert.")
        raise _ToolkitError(msg)


    ## Fit the scoring model.
    ## ----------------------
    if verbose: logger.info("Fitting scoring model....")

    gbt = _gl.boosted_trees_classifier.create(training_accounts,
                                     target='conversion_status',
                                     features=gbt_features,
                                     verbose=verbose, **kwargs)


    ## Fit a segmentation/explanation model using the fitted values from the
    #  predictive model.
    ## ------------------------------------------------------------------------
    if verbose: logger.info("\nFitting segmentation model....")

    training_accounts['conversion_prob'] = gbt.predict(training_accounts,
                                                     output_type='probability')

    max_depth = _max_leaves_to_max_depth(max_segments)

    tree = _gl.decision_tree_regression.create(training_accounts,
                                               target='conversion_prob',
                                               features=gbt.features,
                                               max_depth=max_depth,
                                               verbose=verbose)


    ## Score the open accounts according to the scoring model.
    ## -------------------------------------------------------
    if len(open_accounts) > 0:
        open_accounts = open_accounts.remove_columns(['conversion_status'])

        open_accounts['conversion_prob'] = gbt.predict(open_accounts,
                                                     output_type='probability')

    else:
        logger.warning("There no open accounts to score.")


    ## Segment the training accounts and compute segment statistics.
    ## -------------------------------------------------------------
    training_accounts['segment_id'] = _assign_accounts_to_segments(tree,
                                                             training_accounts)

    open_accounts['segment_id'] = _assign_accounts_to_segments(tree,
                                                               open_accounts)

    segment_descriptions = _compute_segment_stats(tree, training_accounts)


    ## Remove extraneous columns from the outputs.
    ## -------------------------------------------
    for c in [open_duration_name, close_date_name]:
        try:
            open_accounts = open_accounts.remove_column(c)
        except:
            pass

        try:
            training_accounts = training_accounts.remove_column(c)
        except:
            pass

    try:
        open_accounts = open_accounts.remove('decision_date')
    except:
        pass


    ## Construct the LeadScoringModel and return.
    ## ------------------------------------------
    state = {
        'verbose': verbose,
        'account_schema': account_schema,
        'interaction_schema': interaction_schema,
        'trial_duration': trial_duration,
        'max_segments': max_segments,
        'invalid_accounts': invalid_accounts,
        'num_accounts': accounts.num_rows(),
        'num_successes': num_successes,
        'num_explicit_failures': num_explicit_failures,
        'num_implicit_failures': num_implicit_failures,
        'num_open_accounts': num_open_accounts,
        'num_training_accounts': num_training_accounts,
        'num_interactions': num_interactions,
        'num_account_features': len(account_schema['features']),
        'num_interaction_features': num_interaction_features,
        'num_final_features': len(gbt_features),
        'final_features': gbt.features,
        'scoring_model': gbt,
        'segmentation_model': tree,
        'training_account_scores': training_accounts,
        'open_account_scores': open_accounts,
        'segment_descriptions': segment_descriptions,
        'training_time': _time.time() - start_time,
        }

    return LeadScoringModel(state)


## MODEL DEFINITION
## ----------------------------------------------------------------------------

class LeadScoringModel(_CustomModel, _ProxyBasedModel):
    """
    A trained LeadScoringModel estimates the probability that a sales
    account will convert to a specified stage of a marketing or sales funnel,
    based on training data with examples of accounts that have already
    converted and those that have failed to convert.

    This model should not be constructed directly. Instead use the `create`
    method in this module to create and train an instance of this model.
    """
    _PYTHON_LEAD_SCORING_MODEL_VERSION = 0

    required_account_keys = set(['conversion_status', 'account_id'])

    valid_account_keys = {'account_id': (str, int),
                          'account_name': (str,),
                          'features' : (int, float, str, dict, _array.array),
                          'conversion_status': (int,),
                          'decision_date': (_dt.datetime,),
                          'open_date': (_dt.datetime,)}

    required_interaction_keys = set(['account_id'])

    valid_interaction_keys = {'user': (str, int),
                              'account_id': (str, int),
                              'item': (str, int),
                              'features': (int, float, str, dict, _array.array)}

    def __init__(self, state):
        self.__proxy__ = _PythonProxy(state)

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
        >>> model = graphlab.lead_scoring.create(accounts, account_schema)
        >>> model.save('my_model')
        >>> loaded_model = graphlab.load_model('my_model')
        """
        state = self.__proxy__
        pickler.dump(state)

    @classmethod
    def _load_version(self, unpickler, version):
        """
        A function to load a previously created LeadScoringModel instance.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handler.

        version : int
            Version number maintained by the class writer.
        """
        state = unpickler.load()
        return LeadScoringModel(state)

    def _get_summary_struct(self):
        """
        Return a structured description of the model. This includes (where
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
            ('Number of accounts', 'num_accounts'),
            ('Number of interactions', 'num_interactions'),
            ('Number of account features', 'num_account_features'),
            ('Number of interaction features', 'num_interaction_features'),
            ('Verbose', 'verbose'),
            ]

        training_fields = [
            ('Number of training accounts', 'num_training_accounts'),
            ('Number of open accounts', 'num_open_accounts'),
            ('Number of successful conversions', 'num_successes'),
            ('Number of explicit failures', 'num_explicit_failures'),
            ('Number of implicit failures', 'num_implicit_failures'),
            ('Number of final features', 'num_final_features'),
            ('Total training time (seconds)', 'training_time'),
            ]

        section_titles = ["Model schema", "Training summary"]
        return([model_fields, training_fields], section_titles)

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """
        width = 40
        sections, section_titles = self._get_summary_struct()

        accessible_fields = _OrderedDict()
        accessible_fields['open_account_scores'] = "Lead scores, segment IDs, and final features for open accounts."
        accessible_fields['training_account_scores'] = "Lead scores, segment IDs, and final features for training accounts."
        accessible_fields['segment_descriptions'] = "Statistics about market segments of training (i.e. closed) accounts."
        accessible_fields['scoring_model'] = "Underlying GBT model for predicting account conversion."
        accessible_fields['segmentation_model'] = "A trained decision tree to create account segments."
        accessible_fields['account_schema'] = "Schema for the 'accounts' input."
        accessible_fields['interaction_schema'] = "Schema for the 'interactions'SFrame (if provided)."

        out = _toolkit_repr_print(self, sections, section_titles, width=width)
        out2 = _summarize_accessible_fields(accessible_fields, width=width)
        return out + "\n" + out2

    def __str__(self):
        return self.__repr__()

    def get_current_options(self):
        """
        Return the options used to create the current
        :py:class:`~.LeadScoringModel` instance.
        """
        return {k: self.__proxy__[k] for k in get_default_options()['name']}

    def resize_segmentation_model(self, max_segments):
        """
        Change the number of segments created by the model. The lead scoring
        tool's segmentation model is a decision tree, trained on the fitted
        values of the prediction model (a gradient boosted trees model). By
        default, this segmentation model creates 8 segment definitions.
        Reducing this number leads to coarser segments, but a more clear visual
        model representation. Increasing the number produces more homogeneous
        segments, but a more complicated model visualization.

        Parameters
        ----------
        max_segments : int
            Maximum number of segments to create with the tool's segmentation
            model. Because the segmentation model is a binary tree, the actual
            number of segments will likely differ slightly from the argument to
            this parameter.

        Returns
        -------
        LeadScoringModel
            A new instance of `LeadScoringModel`. The original model is
            unchanged.

        Examples
        --------
        >>> accounts = graphlab.SFrame({'account_name': ['Acme', 'Duff', 'Oscorp', 'Tyrell'],
        ...                             'account_id': [0, 1, 2, 3],
        ...                             'conversion_status': [1, -1, 1, 0],
        ...                             'num_employees': [100, 250, 3000, 1132],
        ...                             'product_usage': [1327, 554, 87121, 12755]})
        ...
        >>> model = graphlab.lead_scoring.create(accounts, max_segments=2)
        >>> model2 = model.resize_segmentation_model(max_segments=10)
        """
        _mt._get_metric_tracker().track('toolkit.lead_scoring.resize_segmentation_model')

        ## Convert the max_segments input in the max depth for the segmentation
        #  tree.
        max_depth = _max_leaves_to_max_depth(max_segments)

        if max_depth == self.segmentation_model.max_depth:
            msg = ("Segmentation model is already configured for " +
                   "max_segments of {}.").format(max_segments)
            _logging.warning(msg)

        ## Fit a new segmentation tree and use it to populate a new instance of
        #  LeadScoringModel.
        tree = _gl.decision_tree_regression.create(self.training_account_scores,
                                       target='conversion_prob',
                                       features=self.final_features,
                                       max_depth=max_depth, verbose=False)

        ## Segment the accounts and compute segment statistics
        #  according to the new segmentation model.
        training_accounts = _copy.copy(self.training_account_scores)
        open_accounts = _copy.copy(self.open_account_scores)

        training_accounts['segment_id'] = _assign_accounts_to_segments(tree,
                                                             training_accounts)

        open_accounts['segment_id'] = _assign_accounts_to_segments(tree,
                                                                 open_accounts)

        segment_descriptions = _compute_segment_stats(tree, training_accounts)

        ## Create and return the new model.
        new_state = {k: v for k, v in self.__proxy__.state.items()}
        new_state['segmentation_model'] = tree
        new_state['training_account_scores'] = training_accounts
        new_state['open_account_scores'] = open_accounts
        new_state['segment_descriptions'] = segment_descriptions

        return LeadScoringModel(new_state)

    ## Properties of a trained model
    ## -----------------------------
    @property
    def num_accounts(self):
        """Number of unique accounts in the training 'accounts' dataset."""
        # return self._num_accounts
        return self.__proxy__['num_accounts']

    @property
    def num_successes(self):
        """Number of successful conversions in the training 'accounts'
        dataset."""
        return self.__proxy__['num_successes']

    @property
    def num_explicit_failures(self):
        """Number of accounts explicitly described as failed conversions."""
        return self.__proxy__['num_explicit_failures']

    @property
    def num_implicit_failures(self):
        """Number of accounts determined by the model to have failed
        implicitly."""
        return self.__proxy__['num_implicit_failures']

    @property
    def num_open_accounts(self):
        """Number of open accounts in the 'accounts' dataset. These are the
        accounts that need to be scored by the model."""
        return self.__proxy__['num_open_accounts']

    @property
    def num_training_accounts(self):
        """
        Number of training accounts in the 'accounts' dataset. These are the
        accounts that have already converted successfully or failed, plus open
        accounts that have been determined by the model to have failed
        implicitly because they've been open for too long.
        """
        return self.__proxy__['num_open_accounts']

    @property
    def num_interactions(self):
        """Number of entries in the training 'interactions' dataset."""
        return self.__proxy__['num_interactions']

    @property
    def num_account_features(self):
        """Number of metadata features specified in the 'accounts' table."""
        return self.__proxy__['num_account_features']

    @property
    def num_interaction_features(self):
        """Number of features specified in the 'interactions' table, if
        provided."""
        return self.__proxy__['num_interaction_features']

    @property
    def num_final_features(self):
        """Number of features used in the ultimate prediction model. This count
        includes all features engineered from the interaction data, if
        provided."""
        return self.__proxy__['num_final_features']

    @property
    def training_time(self):
        """Time to train the lead scoring model, in seconds."""
        return self.__proxy__['training_time']

    @property
    def final_features(self):
        """List of all features used in the final prediction and segmentation
        models."""
        return self.__proxy__['final_features']

    @property
    def training_account_scores(self):
        """
        Model output for the training accounts. This SFrame includes the
        estimated conversion probability (i.e. fitted value) for each training
        account, along with a market segment assignment, and the final features
        used by the internal prediction and segmentation models.

        Notes
        -----
        - Training accounts are ones that have already successfully converted
          or failed, or been flagged by the model as failing implicity by
          virtue of being open for too long.

        Examples
        --------
        >>> accounts = graphlab.SFrame({'name': ['Acme', 'Duff', 'Oscorp', 'Tyrell'],
        ...                             'id': [0, 1, 2, 3],
        ...                             'status': [1, -1, 1, 0],
        ...                             'num_employees': [100, 250, 3000, 1132],
        ...                             'product_usage': [1327, 554, 87121, 12755]})
        ...
        >>> account_schema = {
        ...     'conversion_status': 'status',
        ...     'account_id': 'id',
        ...     'account_name': 'name',
        ...     'features': ['num_employees', 'product_usage']}
        ...
        >>> model = graphlab.lead_scoring.create(accounts, account_schema)
        ...
        >>> print model.training_account_scores
        +-----------+--------------+--------------------+---------------+
        | Iteration | Elapsed Time | Training-max_error | Training-rmse |
        +-----------+--------------+--------------------+---------------+
        | 1         | 0.000906     | 0.240460           | 0.223908      |
        +-----------+--------------+--------------------+---------------+
        +------------+-------------------+--------------+---------------+---------------+
        | account_id | conversion_status | account_name | num_employees | product_usage |
        +------------+-------------------+--------------+---------------+---------------+
        |     0      |         1         |     Acme     |      100      |      1327     |
        |     1      |         -1        |     Duff     |      250      |      554      |
        |     2      |         1         |    Oscorp    |      3000     |     87121     |
        +------------+-------------------+--------------+---------------+---------------+
        +---------------------+------------+
        |   conversion_prob   | segment_id |
        +---------------------+------------+
        |  0.8005750775337219 |     2      |
        | 0.28065192699432373 |     1      |
        |  0.8005750775337219 |     2      |
        +---------------------+------------+
        """
        return self.__proxy__['scores']

    @property
    def open_account_scores(self):
        """
        Model output for the open accounts. This SFrame includes the predicted
        conversion probability for each open account, along with a market
        segment assignment, and the final features used by the internal
        prediction and segmentation models.

        Notes
        -----
        - Open accounts are ones that have not yet successfully converted or
          failed. Note that "implicit failures"---accounts determined by the
          model to be failed because they've been open for too long---are
          included in both training and open accounts.

        Examples
        --------
        >>> accounts = graphlab.SFrame({'name': ['Acme', 'Duff', 'Oscorp', 'Tyrell'],
        ...                             'id': [0, 1, 2, 3],
        ...                             'status': [1, -1, 1, 0],
        ...                             'num_employees': [100, 250, 3000, 1132],
        ...                             'product_usage': [1327, 554, 87121, 12755]})
        ...
        >>> account_schema = {
        ...     'conversion_status': 'status',
        ...     'account_id': 'id',
        ...     'account_name': 'name',
        ...     'features': ['num_employees', 'product_usage']}
        ...
        >>> model = graphlab.lead_scoring.create(accounts, account_schema)
        ...
        >>> print model.open_account_scores
        +-----------+--------------+--------------------+---------------+
        | Iteration | Elapsed Time | Training-max_error | Training-rmse |
        +-----------+--------------+--------------------+---------------+
        | 1         | 0.000916     | 0.240460           | 0.223908      |
        +-----------+--------------+--------------------+---------------+
        +------------+--------------+---------------+---------------+--------------------+
        | account_id | account_name | num_employees | product_usage |  conversion_prob   |
        +------------+--------------+---------------+---------------+--------------------+
        |     3      |    Tyrell    |      1132     |     12755     | 0.8005750775337219 |
        +------------+--------------+---------------+---------------+--------------------+
        +------------+
        | segment_id |
        +------------+
        |     2      |
        +------------+
        """
        return self.__proxy__['segment_assignments']

    @property
    def segment_descriptions(self):
        """
        Descriptions and statistics about market segments. The statistics are
        computed using only the training accounts, i.e existing successes or
        failures.

        Examples
        --------
        >>> accounts = graphlab.SFrame({'name': ['Acme', 'Duff', 'Oscorp', 'Tyrell'],
        ...                             'id': [0, 1, 2, 3],
        ...                             'status': [1, -1, 1, 0],
        ...                             'num_employees': [100, 250, 3000, 1132],
        ...                             'product_usage': [1327, 554, 87121, 12755]})
        ...
        >>> account_schema = {
        ...     'conversion_status': 'status',
        ...     'account_id': 'id',
        ...     'account_name': 'name',
        ...     'features': ['num_employees', 'product_usage']}
        ...
        >>> model = graphlab.lead_scoring.create(accounts, account_schema)
        ...
        >>> print model.segment_descriptions
        +------------+----------------------+-----------------------+---------------------+
        | segment_id | mean_conversion_prob | stdev_conversion_prob | min_conversion_prob |
        +------------+----------------------+-----------------------+---------------------+
        |     2      |    0.800575077534    |          0.0          |    0.800575077534   |
        |     1      |    0.280651926994    |          0.0          |    0.280651926994   |
        +------------+----------------------+-----------------------+---------------------+
        +-----------------------+------------------------+---------------------+
        | num_training_accounts | median_conversion_prob | max_conversion_prob |
        +-----------------------+------------------------+---------------------+
        |           2           |     0.800575077534     |    0.800575077534   |
        |           1           |     0.280651926994     |    0.280651926994   |
        +-----------------------+------------------------+---------------------+
        +--------------------------+
        |     segment_features     |
        +--------------------------+
        | [product_usage >= 940.5] |
        | [product_usage < 940.5]  |
        +--------------------------+
        """
        return self.__proxy__['segment_stats']

    @property
    def invalid_accounts(self):
        """
        Input accounts that are invalid because the conversion date occurs
        before the open date.
        """
        return self.__proxy__['invalid_accounts']


