import math as _math
import random as _random
import six as _six

import graphlab as _gl
from graphlab import aggregate as _aggregate
from .. import _sframe_base_view
from graphlab.util import _raise_error_if_not_of_type
from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits._model import Model as _Model
from graphlab.toolkits._internal_utils import \
        _raise_error_if_not_sframe, _raise_error_if_column_exists

from graphlab.deps import numpy as _np


_NoneType = type(None)

class RecommenderEvaluateView(_sframe_base_view.SFrameBaseView):

    def __init__(self, validation_set, models):
        """
        We recommend not constructing this object directly, but instead
        using the method model.views.explore().

        """
        super(RecommenderEvaluateView, self).__init__(
            name = 'Recommender Evaluate',
            label = 'Evaluate',
            tag_name = 'gl-recommender-evaluate',
            description = 'Evaluate view for recommender models'
        )
        if not isinstance(models, dict):
            raise TypeError("Expected the models argument to be a dict.")
        if len(models) == 0:
            raise ValueError("Expected the models argument to non-empty.")
        for v in models.values():
            if not isinstance(v, _Model):
                raise TypeError("Expected the models argument to contain RecommenderModel values")

        # Get metadata from first model
        example_model = list(models.values())[0]
        self._user_column = example_model.user_id
        self._item_column = example_model.item_id
        self._item_counts = example_model.get_num_users_per_item()
        self._user_counts = example_model.get_num_items_per_user()

        self._models = models
        self._validation = validation_set.__copy__()
        cutoffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]
        self.cutoffs = cutoffs
        self._pr_results = self._evaluate_precision_recall(cutoffs)

    def _evaluate_precision_recall(self, cutoffs=[5, 10]):
        results = {}
        for model_name, model in _six.iteritems(self._models):
            result = model.evaluate_precision_recall(self._validation, cutoffs=cutoffs)
            results[model_name] = result
        return results

    def precision_recall(self, cutoffs=[5, 10]):
        """
        For each model, compute the mean precision and recall at the provided cutoffs.

        Parameters
        ----------
        cutoffs : list, optional


        See Also
        --------
        graphlab.evaluation.precision

        Example
        -------
        {'my_model': {'precision_recall': Columns:
                cutoff  int
                precision       float
                recall  float

        Rows: 2

        Data:
        +--------+----------------+--------+
        | cutoff |   precision    | recall |
        +--------+----------------+--------+
        |   5    | 0.333333333333 | 0.375  |
        |   10   | 0.333333333333 | 0.375  |
        +--------+----------------+--------+
        [2 rows x 3 columns]
        }, 'baseline': {'precision_recall': Columns:
                cutoff  int
                precision       float
                recall  float

        Rows: 2

        Data:
        +--------+----------------+--------+
        | cutoff |   precision    | recall |
        +--------+----------------+--------+
        |   5    | 0.333333333333 | 0.375  |
        |   10   | 0.333333333333 | 0.375  |
        +--------+----------------+--------+
        [2 rows x 3 columns]
        }}
        """
        results = self._pr_results
        if not set(cutoffs).issubset(set(self.cutoffs)):
            results = self.evaluate_precision_recall(cutoffs)

        out = {}
        for model_name, result in _six.iteritems(results):
            r = result['precision_recall_overall'][['cutoff', 'precision', 'recall']]
            r = r.filter_by(cutoffs, 'cutoff')
            out[model_name] = {'precision_recall': r}
        return out

    def summary_by_precision(self, cutoffs=[5, 10], num_users_to_sample=20):
        """
        Create a histogram of precision values across users.

        Parameters
        ----------
        cutoffs : list, optional

        num_users_to_sample : int, optional
           For each cutoff and each bin of precision scores, sample the
           provided number of users.

        See Also
        --------
        graphlab.evaluation.precision
        """

        results = self._pr_results
        if not set(cutoffs).issubset(set(self.cutoffs)):
            results = self.evaluate_precision_recall(cutoffs)

        num_items = self._item_counts.num_rows()

        out = {}
        for model_name, result in _six.iteritems(results):
            by_user = result['precision_recall_by_user'].filter_by(cutoffs, 'cutoff')
            by_user['num_recs'] = by_user['cutoff'].apply(lambda x: min(x, num_items))
            by_user['num_correct'] = (by_user['precision'] * by_user['num_recs']).astype(int)
            by_user['precision'] = by_user['precision'].apply(lambda x: _math.floor(x * 100) / 100)
            by_user['recall'] = by_user['recall'].apply(lambda x: _math.floor(x * 100) / 100)
            cols = ['cutoff', 'num_correct', 'precision', 'recall']
            by_user =  by_user.groupby(cols, {'num_users': _aggregate.COUNT,
                                              'random_user_sample': _aggregate.CONCAT(self._user_column)})
            by_user['random_user_sample'] = by_user['random_user_sample'].apply(lambda x: x[:num_users_to_sample], dtype=list)
            out[model_name] = by_user.sort(['cutoff', 'precision'])
        return out


    def summary_by_activity(self, cutoffs=[5, 10], num_bins=20, logscale=False):
        results = self._evaluate_precision_recall(cutoffs)

        # Get maximum value
        max_value = self._user_counts['num_items'].max() * 1.0
        if logscale:
          max_value = _math.log(max_value)

        out = {}
        for model_name, result in results.items():
            by_user = result['precision_recall_by_user'].filter_by(cutoffs, 'cutoff')

            # Get number of training observations for each user
            by_user = by_user.join(self._user_counts, on=self._user_column)
            by_user.rename({'num_items': 'num_obs'})

            if logscale:
                by_user['num_obs'] = by_user['num_obs'].apply(lambda x: _math.log(x))

            # Assign users to bins based on a histogram of activity.
            bin_id = by_user['num_obs'] / max_value
            bin_id = bin_id.clip_upper(0.9999) * num_bins
            bin_id = bin_id.astype(int)
            by_user['bin'] = bin_id

            # Get information about bins
            bins = _gl.SFrame({'bin': list(range(num_bins))})
            bin_width =  max_value / num_bins
            bins['bin_start'] = bins['bin'] * bin_width
            bins['bin_end'] = (bins['bin'] + 1) * bin_width

            # For each cutoff, create summaries of each bin.
            aggs = {'min_num_observations': _gl.aggregate.MIN('num_obs'),
                    'num_users': _gl.aggregate.COUNT(),
                    'mean_precision': _gl.aggregate.MEAN('precision'),
                    'std_precision': _gl.aggregate.STDV('precision'),
                    'mean_recall': _gl.aggregate.MEAN('recall'),
                    'std_recall': _gl.aggregate.STDV('recall')}
            by_user = by_user.groupby(['cutoff', 'bin'], aggs).sort(['cutoff', 'min_num_observations'])
            by_user = by_user.join(bins, on='bin', how='left')

            # Note that the num_users column should match up with hist.
            out[model_name] = {'precision_recall': by_user}

        return out


    def validation_summary(self):
        results = self._validation.groupby(self._user_column, {'num_interactions': _aggregate.COUNT})
        return {'num_users': results.num_rows(),
                'mean_interactions_per_user': results['num_interactions'].mean()}
