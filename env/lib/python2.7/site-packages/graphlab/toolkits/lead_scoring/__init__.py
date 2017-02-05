"""
The GraphLab Create Lead Scoring toolkit estimates the probability that an open
sales account account will successfully convert to a given stage of the
marketing or sales funnel. It also segments the training accounts, i.e. those
have already converted or failed, into similar groups. The features that
describe these groups can be used to define new marketing campaigns.

While traditional lead scoring tools simply assign points for each interaction
between an open account and your business, the GraphLab Create tool uses
state-of-the-art probabilistic modeling to automatically determine which
account features are most predictive of a successful conversion.

The fundamental input to the lead scoring toolkit is a table of information
about accounts. The conversion status field indicates if an account has
successfully converted already (1), failed (-1), or remains open and needs to
be predicted (0).

.. sourcecode:: python

    >>> accounts = graphlab.SFrame({'name': ['Acme', 'Duff', 'Oscorp', 'Tyrell'],
    ...                             'status': [1, -1, 1, 0],
    ...                             'num_employees': [100, 250, 3000, 1132],
    ...                             'product_usage': [1327, 554, 87121, 12755]})
    ...
    >>> account_schema = {'conversion_status': 'status',
    ...                   'account_name': 'name',
    ...                   'features': ['num_employees', 'product_usage']}
    ...
    >>> model = graphlab.lead_scoring.create(accounts, account_schema)

A trained lead scoring model contains three primary items of interest:

    - `open_account_scores`: predicted conversion probability and market
      segment assignment for each *open* account.

    - `training_account_scores`: estimated conversion probability (i.e. fitted
      value) and market segment assignment for each training account (i.e.
      accounts that have already failed or successfully converted).

    - `segment_descriptions`: feature descriptions and summary statistics for
      market segments, based on the training accounts.

.. sourcecode:: python

    >>> print model.open_account_scores
    +--------------+---------------+---------------+-----------------+------------+
    | account_name | num_employees | product_usage | conversion_prob | segment_id |
    +--------------+---------------+---------------+-----------------+------------+
    |    Tyrell    |      1132     |     12755     |  0.800575077534 |     2      |
    +--------------+---------------+---------------+-----------------+------------+
    ...
    >>> print model.training_account_scores
    +-------------------+--------------+---------------+---------------+
    | conversion_status | account_name | num_employees | product_usage |
    +-------------------+--------------+---------------+---------------+
    |         1         |     Acme     |      100      |      1327     |
    |         -1        |     Duff     |      250      |      554      |
    |         1         |    Oscorp    |      3000     |     87121     |
    +-------------------+--------------+---------------+---------------+
    +-----------------+------------+
    | conversion_prob | segment_id |
    +-----------------+------------+
    |  0.800575077534 |     2      |
    |  0.280651926994 |     1      |
    |  0.800575077534 |     2      |
    +-----------------+------------+
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

If available, data about how each account interactions with your business
assets (e.g. products, websites, email campaigns) can also be passed to the
lead scoring model, through the `interactions` parameter. The lead scoring tool
automatically creates account-level features from this interaction data.

If this data is provided, a bit more metadata is needed about each account so
the model can create relevant features correctly. In particular, accounts need
an ID, an "open date" (date on which the account was created), and a "decision
date" (date on which the conversion status was decided, for training accounts
only). We also need to choose the 'trial_duration', which determines the date
range for the usage-based feature engineering.

.. sourcecode:: python

    >>> import datetime as dt
    >>> accounts['id'] = [0, 1, 2, 3]
    >>> accounts['open_date'] = [dt.datetime(2016, 6, 1),
    ...                          dt.datetime(2016, 6, 1),
    ...                          dt.datetime(2016, 6, 15),
    ...                          dt.datetime(2016, 6, 23)]
    >>> accounts['converted_date'] = [dt.datetime(2016, 6, 10),
    ...                               dt.datetime(2016, 6, 20),
    ...                               dt.datetime(2016, 6, 20),
    ...                               None]
    ...
    >>> account_schema['account_id'] = 'id'
    >>> account_schema['open_date'] = 'open_date'
    >>> account_schema['decision_date'] = 'converted_date'
    ...
    >>> interactions = graphlab.SFrame({
    ...     'id': [0, 0, 1, 1, 3, 3, 3],
    ...     'item': ['a', 'b', 'a', 'a', 'a', 'c', 'c'],
    ...     'user': [13, 13, 17, 19, 23, 23, 29],
    ...     'item_version': ['1.1', '1.2', '1.1', '1.2', '1.1', '1.3', '1.1'],
    ...     'timestamp': [dt.datetime(2016, 6, 2), dt.datetime(2016, 6, 3),
    ...                   dt.datetime(2016, 6, 4), dt.datetime(2016, 6, 5),
    ...                   dt.datetime(2016, 6, 24), dt.datetime(2016, 6, 24),
    ...                   dt.datetime(2016, 6, 25)]})
    ...
    >>> interactions = graphlab.TimeSeries(interactions, index='timestamp')
    ...
    >>> interaction_schema = {'account_id': 'id', 'user': 'user', 'item': 'item'}
    ...
    >>> model = graphlab.lead_scoring.create(accounts, account_schema,
    ...                                      interactions, interaction_schema,
    ...                                      trial_duration=dt.timedelta(days=30))

.. warning:: The lead scoring toolkit is in beta, and may change in future
    versions of GraphLab Create.
"""

from ._lead_scoring import create
from ._lead_scoring import LeadScoringModel
from ._lead_scoring import get_default_options


