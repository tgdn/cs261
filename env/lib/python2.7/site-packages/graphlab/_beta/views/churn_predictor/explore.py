from .. import _sframe_base_view
import graphlab as _gl
from .util import _get_churn_predictor_model_info

_NUM_SAMPLE_USERS = 6
def _get_random_users(user_list):
    import random
    random.seed(100)
    K = min(_NUM_SAMPLE_USERS, len(user_list))
    return random.sample(user_list, K)

def _get_flattend_sframe(data, primary_key, data_col_name = 'data'):
    cols = data.column_names()
    cols.remove(primary_key)
    return data.pack_columns(cols, dtype = dict, new_column_name='data')\
               .unstack('data', data_col_name)

def _annotate_global_churn_reasons(text):
    ret = []
    start = 0
    state = "TEXT"
    for i in range(len(text)):
        if text[i] == '"' and state == "TEXT":
            ret += [(text[start:i], state)]
            start = i
            state = "FEATURE"
            continue
        if text[i] == '"' and state == "FEATURE":
            ret += [(text[start:i+1], state)]
            start = i+1
            state = "TEXT"
            continue
    if start != len(text) - 1:
        ret += [(text[start:], state)]
    return ret

class ChurnPredictorExploreView(_sframe_base_view.SFrameBaseView):

    def __init__(self, model, observation_data, time_boundary = None,
                 user_data = None):
        super(ChurnPredictorExploreView, self).__init__(
            name = 'Churn Predictor Explore',
            label = 'Explore',
            tag_name = 'gl-churn-predictor-explore',
            description = 'Explore view for ChurnPredictor models.'
        )

        # Extract features from the boosted trees.
        # -----------------------------------------------------------------
        if time_boundary is None:
            time_boundary = observation_data.max_time
        model_info = _get_churn_predictor_model_info(model, time_boundary)
        predictions = model._internal_predict(observation_data,
                time_boundary=time_boundary, user_data=user_data,
                include_leaf_id = True)

        # Segment predictions.
        raw_segments = model._get_segments_from_predictions(predictions,
                                                      max_segments=20)

        # Aggregate segments
        segments = model._prepare_churn_report(raw_segments, time_boundary, True)
        segments = segments.rename({'num_users': 'count',
                         'num_users_percentage': 'percentage'})
        segments['is_other'] = segments['segment_info'].apply(
                                              lambda x: len(x) > 1)
        segments['sample_users'] = segments['users'].apply(_get_random_users, list)
        segments = segments.remove_column('users')

        # Get the time-series and user-data associated with some sample users.
        user_id = model.user_id
        import itertools
        all_sample_users = list(itertools.chain(*list(segments['sample_users'])))

        # Get the interaction data
        interaction_data = observation_data.filter_by(all_sample_users, user_id)
        sample_user_data = _get_flattend_sframe(interaction_data.to_sframe(),
                user_id, 'interaction_data')

        # Get the timeline data.
        import graphlab as gl
        interaction_data_grp = gl.GroupedTimeSeries(interaction_data, user_id)
        timeline_data = gl.SFrame()
        for uid, user_ts in interaction_data_grp:
            resampled_user_ts = user_ts.resample(model.time_period,
                                                gl.aggregate.COUNT(), "zero")
            resampled_user_ts = resampled_user_ts[resampled_user_ts['Count'] > 0]
            resampled_user_ts[user_id] = uid
            timeline_data = timeline_data.append(resampled_user_ts.to_sframe())
        timeline_data = _get_flattend_sframe(timeline_data, user_id,
                'timeline_data')
        sample_user_data = sample_user_data.join(timeline_data, user_id)

        # Get the side-data
        if user_data:
            sample_users_sd = _get_flattend_sframe(
                user_data.filter_by(all_sample_users, user_id),
                user_id, 'user_data')
            sample_user_data = sample_user_data.join(sample_users_sd, user_id)

        # Get the reasons for churn & probability.
        user_explanations = segments[['segment_id', 'explanation', 'sample_users',
            'annotated_explanation']].stack('sample_users', new_column_name = user_id)
        sample_user_data = sample_user_data.join(user_explanations, user_id)\
                 .join(predictions[[user_id, 'probability']], user_id)

        sample_user_data['user_id'] = sample_user_data[user_id]
        if user_id != 'user_id':
            sample_user_data.remove_column(user_id)

        # Get data info.
        # -----------------------------------------------------------------
        data_info = {
           'num_users'       : len(predictions),
           'num_observations': len(observation_data)
        }

        # Raw Distribution of Prediction Probabilities rolled up
        # -----------------------------------------------------------------
        prediction_distributions = self._get_distribution(predictions, num_bins=50)
        density_estimate = self._get_density_estimate(prediction_distributions['distribution'])

        segments = segments.sort('avg_probability', ascending = False)

        # Top reasons to churn.
        # -----------------------------------------------------------------
        self._top_churn_reasons = \
                list(model.get_feature_importance()['description'])
        self._top_churn_reasons_annotated = \
               list(map(_annotate_global_churn_reasons, self._top_churn_reasons))

        # Cache things.
        # -----------------------------------------------------------------
        self._model_info = model_info
        self._data_info = data_info
        self._distribution = prediction_distributions
        self._segments = segments
        self._segment_index = {
           s['segment_id']:row for row, s in enumerate(self._segments)
        }
        raw_segments['user_id'] = raw_segments[user_id]
        raw_segments = raw_segments[['user_id', 'probability', 'segment_id']]
        self._user_list = {
           sid: raw_segments[raw_segments['segment_id'] == sid] \
                                               for sid in self._segment_index
        }
        self._raw_segments = raw_segments
        self._density_estimate = density_estimate
        self._predictions = predictions
        self._timestamp_col_name = observation_data.index_col_name
        self._sample_user_data = {}
        for user in sample_user_data:
            segment_id = user['segment_id']
            user_id = user['user_id']
            if not(segment_id in self._sample_user_data):
                self._sample_user_data[segment_id] = {}
            self._sample_user_data[segment_id][user_id] = user

    def get_timestamp_col_name(self):
        return self._timestamp_col_name

    def get_kde_estimation(self, bandwidth=None, num_bins=None):
        '''
        return 200 numbers to estimate the raw distribution of predicted probability, return type is SFrame

        Example Output
          +------------------+--------------+
          |       x          |  density     |
          +------------------+--------------+
          |       0.000      |   0.01445    |
          |       0.005      |   0.06598    |
          |       0.010      |   0.07591    |
          |       0.015      |   0.08348    |
          |       0.020      |   0.06482    |
          |       0.025      |   0.0448     |
          |       0.030      |   0.09123    |
          |       0.035      |   0.09432    |
          |       0.040      |   0.09843    |
          |       0.045      |   0.1024     |
          |       0.050      |   0.1120     |
          +------------------+--------------+
          [200 rows x 2 columns]
        '''

        if num_bins is not None:
            self._distribution = self._get_distribution(self._predictions, num_bins=num_bins)
        if bandwidth is not None:
            self._density_estimate = self._get_density_estimate(predictions=self._distribution['distribution'], bandwidth=bandwidth)
        return self._density_estimate

    def _get_density_estimate(self, predictions, bandwidth=0.1):
        '''
        compute 200 numbers to represent the distribution of the predicted probability
        The output x and y are scaled to the range of [0, 1]
        '''
        output_size = 200.0

        def epanechnikov(u):
            if u > 1 or u < -1:
                return 0.0
            else:
                return 0.75 - 0.75 * u * u
        # use binned data to approximate density
        probability = predictions['mean_probability']
        weight = predictions['count']
        total_weight = predictions['count'].sum()

        x_val = _gl.SArray(range(int(output_size)))
        x_val = x_val / output_size # [0.001, 0.002, ..., 0.999]

        estimate = []
        for i, x in enumerate(x_val):
            est =  0
            for j, prob in enumerate(probability):
                est += weight[j] * epanechnikov( (x - prob) / bandwidth )
            est = est / total_weight / bandwidth
            estimate.append(est)

        # then scale the estimate to [0, 1] range
        estimate = _gl.SArray(estimate)
        est_max = estimate.max()
        est_min = min(estimate.min(), 0)
        if (est_max - est_min) > 0:
            estimate = (estimate - float(est_min)) / float(est_max - est_min)
        else:
            # min == max, it is a constant signal
            estimate = estimate * 0 + 0.5 # make it [0.5, ..., 0.5]
        ret = _gl.SFrame()
        ret['x'] = x_val
        ret['density'] = estimate
        return ret

    def _get_distribution(self, data, num_bins = 100):
        num_predictions = len(data)

        data['bin_id'] = data['probability'] * num_bins
        data['bin_id'] = data['bin_id'].astype(int)

        dist = data.groupby('bin_id', {
            'count'       : _gl.aggregate.COUNT(),
            'mean_probability' : _gl.aggregate.MEAN('probability')
        })
        dist['fraction'] = dist['count'] / num_predictions
        dist, no_events = dist.dropna_split()
        dist = dist.sort('mean_probability')
        if (no_events.num_rows() == 0):
            no_events = no_events.append(_gl.SFrame({
                "bin_id": [0],
                "count": [0],
                "fraction": [0.00],
                "mean_probability": [0.00],
            }))

        return {
            "distribution": dist['mean_probability', 'count', 'fraction'],
            "no_events": no_events['count', 'fraction']
        }

    def get_distribution(self):
        """
        Get the distribution of prediction probabilities for the validation data
        and the count of users for which no prediction could be made.

        Returns
        -------
        out: dictionary

        Example Output
        --------------
        { distribtion:
          +------------------+----------+--------------+
          | mean_probability | count    |  fraction    |
          +------------------+----------+--------------+
          |        0.0       |  10      |   0.01445    |
          |       0.01       |  74      |   0.06598    |
          |       0.02       |  62      |   0.07591    |
          |       0.03       |  58      |   0.08348    |
          |       0.04       |  87      |   0.06482    |
          |       0.05       |  48      |   0.0448     |
          |       0.06       |  98      |   0.09123    |
          |       0.07       |  101     |   0.09432    |
          |       0.08       |  104     |   0.09843    |
          |       0.09       |  120     |   0.1024     |
          |       0.10       |  125     |   0.1120     |
          +------------------+----------+--------------+
          [100 rows x 3 columns]

        no_events:
          +----------+--------------+
          | count    |  fraction    |
          +----------+--------------+
          |  10      |   0.01445    |
          +----------+--------------+
        }
        """
        return self._distribution

    def get_observation_data_info(self):
        """
        Get the information about the data.

        Returns
        -------
        out: dict
        A dictionary with all the information about the model.

        Example Output
        --------------
        {
         'num_observations': 99950,
         'num_users': 994,
        }
        """
        return self._data_info

    def get_model_info(self):
        """
        Get the information about the model.

        Returns
        -------
        out: dict
        A dictionary with all the information about the model.

        Example Output
        --------------
        {
            'categorical_features': ['action'],
             'churn_period': 2592000.0,
             'features': ['action'],
                'forecast_end': '1970-05-16 07:23:21',
                'forecast_start': '1970-04-16 07:23:21',
             'is_data_aggregated': 0,
             'lookback_periods': [7, 14, 21, 60, 90],
             'model_options': {'max_depth': 6, 'min_loss_reduction': 0.0},
             'num_features': 1,
             'num_observations': 99950,
             'num_users': 994,
             'numerical_features': [],
             'time_boundaries': ['1970-01-13 18:30:01',
                                 '1970-01-25 08:06:41',
                                 '1970-02-05 21:43:21',
                                 '1970-02-17 11:20:01',
                                 '1970-03-01 00:56:41',
                                 '1970-03-12 14:33:21',
                                 '1970-03-24 04:10:01',
                                 '1970-04-04 17:46:41',
                                 '1970-04-16 07:23:21'],
             'time_period': 86400.0,
             'user_id': 'user_id'
          }
        """
        return self._model_info

    def get_global_churn_reasons(self):
        """
        Get the global influcers for churn vs not-churn.

        Example Output
        --------------
        [
          'Number of 30 day periods since last activity',
          ...
          'Number of decreasing time ... '
        ]
        """
        return self._top_churn_reasons

    def get_annotated_global_churn_reasons(self):
        """
        Get the global influcers for churn vs not-churn.

        Example Output
        --------------
        [
          [('', 'TEXT'),
           ('"Using"', 'FEATURE'),
           (' events in feature ', 'TEXT'),
           ('"action"', 'FEATURE'),
           (' in the last 7 days', 'TEXT')]
          ...
        ]
        """
        return self._top_churn_reasons_annotated


    def get_segment(self, segment_id):
        """
        Get the info of a single segment.

        Parameters
        ----------
        segment_id (string): A segment id (must be present in the data).

        A dictionary which looks like the following:
         - segment_id       : Segment id
         - sample_users     : Sample users in this segment
         - count            : Number of users in this segment.
         - explanation      : Explanation of predictions.
         - annotated_explanation : Explanations with annotations for coloring.
         - percentage       : Percentage number of users in this segment.
         - avg_probability  : Avg prob. of churn for all users in this segment.
         - max_probability  : Max prob.
         - min_probability  : Min prob.
         - stdv_probability : Stdv prob.
         - is_other         : True if its a collapsed segment.
         - segment_info     : Dictionary of sub-segments collapsed into this.
                               - key   : leaf id (internal user)
                               - value : number of users in this sub-segment.
        Example Output
        --------------
        {
           'avg_probability': 0.12995091997660124,
           'count': 13,
           'annotated_explanation':
              [
                 [('Fewer than', 'TEXT'),
                  ('20', 'VALUE'),
                  ('"Using"', 'FEATURE'),
                  (' events in feature ', 'TEXT'),
                  ('"action"', 'FEATURE'),
                  (' in the last 7 days', 'TEXT')]
                 ...
              ]
           'explanation':
              ['Number of days since last activity <= 2.5',
                           ....
              'Longest time between actions in the previous 21 days <= 13'],
           'is_other': 0,
           'max_probability': 0.27886664867401123,
           'min_probability': 0.06382368505001068,
           'percentage': 1.3078470824949697,
           'segment_id': '2',
           'segment_info': {49: 13},
           'stdv_probability': 0.05825092077089031
        }


        """
        if segment_id not in self._segment_index:
            raise KeyError("Segment '%s' does not exist." % segment_id)
        return self._segments[self._segment_index[segment_id]]

    def get_segments(self):
        """
        Get info all segments in the data.

        Example Output
        --------------
        [
           {'avg_probability': 0.12995091997660124,
           'count': 13,
           'explanation':
              ['Number of days since last activity <= 2.5',
                           ....
              'Longest time between actions in the previous 21 days <= 13'],
           'is_other': 0,
           'max_probability': 0.27886664867401123,
           'min_probability': 0.06382368505001068,
           'percentage': 1.3078470824949697,
           'segment_id': '2',
           'segment_info': {49: 13},
           'stdv_probability': 0.05825092077089031},

           {
             'avg_probability': 0.29958346486091614,
             'count': 2,
             'explanation': None,
             'is_other': 1,
             'max_probability': 0.33462783694267273,
             'min_probability': 0.26453909277915955,
             'percentage': 0.2012072434607646,
             'segment_id': 'Other (Active)',
             'segment_info': {50: 1, 64: 1},
             'stdv_probability': 0.03504437208175659
            }
        ]
        """
        return self._segments

    def get_sample_users_in_segment(self, segment_id):
        """
        Get the information about sample users within a segment.

        Get the info of a single segment.

        Parameters
        ----------
        segment_id (string): A segment id (must be present in the data).

        A dictionary which looks like the following:

         - user_id          : User id
         - probability      : Probability of churn
         - segment_id       : Segment id
         - explanation      : Explanation of predictions.
         - interaction_data : All the interaction data involving this user.
         - timeline_data    : The interaction data as a count timeline
         - user_data        : User metadata.

        Example Output
        --------------
         {
          'user_id': 'user-675',
          'segment_id': '6',
          'probability  : 0.75,
          'explanation':
            ['Less than 2.50 days since most recent event',
             'Less than 73.00 "Quitting" events in feature "action" in the last 60 days',
             'Less than 37.50 "Using" events in feature "action" in the last 60 days'
            ],
          'timeline_data': [
              {'Count': '1',
               'timestamp': datetime.datetime(1970, 4, 22, 20, 26, 41)},
              {'Count': '2',
               'timestamp': datetime.datetime(1970, 4, 22, 20, 43, 21)},
              {'Count': '3',
               'timestamp': datetime.datetime(1970, 4, 16, 23, 36, 41)}],
          'interaction_data': [
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 22, 20, 26, 41)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 22, 20, 43, 21)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 22, 22, 6, 41)},
              {'action': 'Trying',
               'timestamp': datetime.datetime(1970, 4, 23, 7, 33, 21)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 23, 10, 3, 21)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 23, 14, 46, 41)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 23, 16, 26, 41)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 23, 19, 30, 1)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 23, 19, 46, 41)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 23, 22, 33, 21)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 23, 23, 56, 41)},
              {'action': 'Quitting',
               'timestamp': datetime.datetime(1970, 4, 24, 3, 33, 21)},
              {'action': 'Trying',
               'timestamp': datetime.datetime(1970, 4, 16, 23, 36, 41)}],
          }
        """
        if segment_id not in self._sample_user_data:
            raise KeyError("Segment '%s' does not exist." % segment_id)
        return self._sample_user_data[segment_id]

    def get_sample_users(self):
        """
        Get all the information about the sample users.
        """
        return self._sample_user_data

    def get_segment_user_list(self, segment_id, max_users = 10000):
        """
        Get all the users (top max_users) that belong to a segment.

        Example Output
        --------------
        +----------+----------------+------------+
        | user_id  |  probability   | segment_id |
        +----------+----------------+------------+
        |  user-0  | 0.977338552475 |     0      |
        |  user-1  | 0.977338552475 |     0      |
        | user-999 | 0.977338552475 |     0      |
        | user-100 | 0.977338552475 |     0      |
        | user-101 | 0.977338552475 |     0      |
        | user-102 | 0.977338552475 |     0      |
        | user-103 | 0.977338552475 |     0      |
        | user-104 | 0.977338552475 |     0      |
        | user-105 | 0.977338552475 |     0      |
        | user-106 | 0.977338552475 |     0      |
        +----------+----------------+------------+
        [938 rows x 3 columns]

        """
        return self._user_list[segment_id].topk('probability', max_users)

    def get_range_user_list(self, start, end, max_users = 10000):
        """
        Get all the users (top max_users) with churn probability [start, end].


        Example Output
        --------------
        +----------+----------------+----------------+
        | user_id  |  probability   |   segment_id   |
        +----------+----------------+----------------+
        | user-885 | 0.392031341791 |       1        |
        | user-253 | 0.375937104225 |       3        |
        | user-655 | 0.334627836943 | Other (Active) |
        | user-281 | 0.325749844313 |       1        |
        | user-516 | 0.278866648674 |       2        |
        | user-769 | 0.264539092779 | Other (Active) |
        | user-357 | 0.219812572002 |       3        |
        | user-589 | 0.208243981004 |       1        |
        | user-708 | 0.181244313717 |       2        |
        | user-292 | 0.181244313717 |       2        |
        +----------+----------------+----------------+
        [16 rows x 3 columns]

        """
        log_filter = (self._raw_segments['probability'] >= start) & \
                     (self._raw_segments['probability'] <= end)
        return self._raw_segments[log_filter].topk('probability', max_users)
