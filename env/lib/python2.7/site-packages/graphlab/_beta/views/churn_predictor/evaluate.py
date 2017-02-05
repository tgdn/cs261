import math as _math
import six as _six
import graphlab as _gl
from .. import _sframe_base_view
from graphlab.util import _raise_error_if_not_of_type
from .util import _get_churn_predictor_model_info

_NoneType = type(None)

class ChurnPredictorEvaluateView(_sframe_base_view.SFrameBaseView):
    def __init__(self, observation_data, time_boundary, user_data = None,
            models = None):
        super(ChurnPredictorEvaluateView, self).__init__(
            name = 'Churn Predictor Evaluate',
            label = 'Evaluate',
            tag_name = 'gl-churn-predictor-evaluate',
            description = 'Evaluate view for churn predictor models'
        )

        _raise_error_if_not_of_type(models, [dict], 'models')
        if len(models) == 0:
            raise ValueError("`models` argument cannot be empty.")
        for v in models.values():
            if not isinstance(v, _gl.churn_predictor.ChurnPredictor):
                raise TypeError("Only churn predictor models can be evaluated.")

        # Get out an example model.
        # TODO: Check that the 2 models were trained on the same user_id.
        churn_model = list(models.values())[0]
        self._user_column = churn_model.user_id
        self._models = models

        # Get model schema, data info, and model predictions.
        # --------------------------------------------------------------------
        model_info = {}
        for model_name, model in _six.iteritems(models):
            model_info[model_name] = _get_churn_predictor_model_info(model,
                    time_boundary)

        # Make predictions
        predictions = {}
        for name, model in _six.iteritems(self._models):
            preds = model._evaluation_prepare(observation_data,
                    time_boundary, user_data)
            preds = preds.dropna('probability')
            predictions[name] = preds

        data_info = {
           'num_users'       : len(predictions[list(self._models.keys())[0]]),
           'num_observations': len(observation_data)
        }

        # Pre-compute the precision and recall at 100 cut-offs.
        # --------------------------------------------------------------------
        results = {}
        for name, model in _six.iteritems(self._models):
            evals =  model._get_metrics_from_predictions(
                predictions[name], ["auc", "precision", "recall", "roc_curve"])
            results[name] = {
               "evaluation_data" : evals['evaluation_data'],
               "auc"             : evals["auc"],
               "precision"       : evals["precision"],
               "recall"          : evals["recall"],
               "roc_curve"       : evals["roc_curve"],
            }
        self._results = results

        num_cutoffs = 100
        cutoffs = [float(x)/num_cutoffs for x in range(num_cutoffs)]
        all_precision_recall = self._evaluate_precision_recall(cutoffs=cutoffs)

        # Create a distribution of predictions.
        # --------------------------------------------------------------------
        evals = self._results['churn_predictor']['evaluation_data']

        distribution = {}
        distribution['churn_predictor'] = churn_dist = self._get_distribution(evals)
        distribution['baseline'] = self._get_distribution(self._results['baseline']['evaluation_data'])

        threshold_counts = self._compute_threshold_counts()

        # Cache things needed for the future.
        # ---------------------------------------------------------------------
        self._distribution = distribution
        self._cutoffs = cutoffs
        self._model_info = model_info
        self._data_info = data_info
        self._all_precision_recall = all_precision_recall
        self._threshold_counts = threshold_counts

    def _compute_threshold_counts(self):
        ret = {}
        for name, results in _six.iteritems(self._results):
            sf = results["roc_curve"].add_row_number("id")
            sf = sf[sf["id"] % 100 == 0]
            # NOTE the evaluate API should provide these raw counts.
            # floating point precision with rounding is a temporary approximation
            # for now.
            sf["tp"] = sf["tpr"] * sf["p"]
            sf["fn"] = sf["p"] - sf["tp"]
            sf["fp"] = sf["fpr"] * sf["n"]
            sf["tn"] = sf["n"] - sf["fp"]
            for col in ["tp", "fn", "fp", "tn"]:
                sf[col] = _gl.SArray(list(map(round, sf[col])), dtype=int)
            ret[name] = sf[['threshold', 'tp', 'fp', 'tn', 'fn']]
        return ret

    def get_threshold_counts(self):
        return self._threshold_counts

    def _get_distribution(self, eval_data):
        num_predictions = len(eval_data)
        num_bins = 10
        bin_size = 1.0 / num_bins

        # assign ids to bins, note this clips 1.0 to 0.999 to prevent creating an id for values of 1.0
        eval_data['bin_id'] = eval_data['probability'].clip_upper(0.999) * num_bins
        eval_data['bin_id'] = eval_data['bin_id'].astype(int)

        dist = eval_data.groupby('bin_id', {
           'count'       : _gl.aggregate.COUNT(),
           'churn_count' : _gl.aggregate.SUM('label'),
           'mean_probability' : _gl.aggregate.MEAN('probability')
         }).join(_gl.SFrame({'bin_id': range(num_bins)}), how='outer')
        dist = dist.fillna('count', 0)
        dist = dist.fillna('churn_count', 0)
        dist['fraction'] = dist['count'] / num_predictions
        dist['churn_fraction'] = \
                dist['churn_count'].astype(float) / dist['count']
        dist = dist.sort('bin_id')

        dist['bin_start'] = dist['bin_id'] * bin_size
        dist['bin_end'] = (dist['bin_id'] + 1) * bin_size
        return dist

    def get_observation_data_info(self):
        """
        Get the information about each model.

        Returns
        -------
        out: dict
        A dictionary with all the information about the data.

        Example Output
        --------------
        {
          'churn_predictor':
              {
               'num_observations': 99950,
               'num_users': 994,
              }
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
        {'churn_predictor':

            {'categorical_features': ['action'],
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
        }
        """
        return self._model_info

    def get_metrics(self):
        '''
        Get all the performance metrics from the model.

        Example output
        --------------
        {
          'precision' : {
              'churn_predictor': 0.5,
              'baseline': 0.5,
          },
          'recall' : {
              'churn_predictor': 0.5,
              'baseline': 0.5,
          },
          'auc' : {
              'churn_predictor': 0.5,
              'baseline': 0.5,
          }
        }
        '''
        all_ret = {}
        for metric in ['auc', 'precision', 'recall']:
            ret = {}
            for model_name, result in _six.iteritems(self._results):
                ret[model_name] = result[metric]
            all_ret[metric] = ret
        return all_ret


    def precision_recall(self):
        """
        For each mode, compute the precision recall curve
        """
        return self._all_precision_recall

    def _precision(self, num_correct_predictions, num_predicted_churn):
        precision = float('nan')
        if num_predicted_churn > 0:
            precision = \
              float(num_correct_predictions) / float(num_predicted_churn)
        return precision

    def _recall(self, num_found_churn, num_total_churn):
        recall = float('nan')
        if num_total_churn > 0:
            recall = float(num_found_churn) / float(num_total_churn)
        return recall

    def _evaluate_precision_recall(self, cutoffs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        '''
        Get the recall of all models along with useful information about it.

        Parameters
        ----------
        cutoffs: list, optional

        Example output
        --------------
        {
            'baseline':
            {
              'total_churn': 189,
              'precision_recall': SFrame(
              +------------+-----------+-------------+----------------------+
              |   recall   | precision | found_churn |  correct_prediction  |
              +------------+-----------+-------------+----------------------+
              |  0.8412698 | 0.6068702 |    159      |       160            |
              |   ....     |   ....    |    ...      |       ...            |
              +------------+-----------+-------------+----------------------+
              +--------+----------------+
              | cutoff | predicted_churn|
              +--------+----------------+
              |   0.1  |      262       |
              |   ...  |      ...       |
              +--------+----------------+
              )
            },
            'churn_predictor':
            {
               'total_churn': 189,
               'precision_recall': SFrame(
               +------------+-----------+-------------+----------------------+
               |   recall   | precision | found_churn |  correct_prediction  |
               +------------+-----------+-------------+----------------------+
               |  0.8412698 | 0.6068702 |    159      |       160            |
               |   ....     |   ....    |    ...      |       ...            |
               +------------+-----------+-------------+----------------------+
               +--------+----------------+
               | cutoff | predicted_churn|
               +--------+----------------+
               |   0.1  |      262       |
               |   ...  |      ...       |
               +--------+----------------+
               )
            }
        }
        '''
        ret = {}
        for model_name, result in _six.iteritems(self._results):
            ret[model_name] = {}
            evaluation_data = result['evaluation_data']
            total_churn = evaluation_data[evaluation_data['label'] == 1]
            num_total_churn = len(total_churn)

            ret[model_name]['total_churn'] = num_total_churn
            recall = []
            precision = []
            predicted_churn = []
            correct_prediction = []
            found_churn = []
            cutoff_collection = []

            for cutoff in cutoffs:

                predicted_churn_users = \
                        evaluation_data[evaluation_data['probability'] >= cutoff]
                correct_prediction_users = predicted_churn_users[predicted_churn_users['label'] == 1]
                num_predicted_churn = len(predicted_churn_users)
                num_correct_churn = len(correct_prediction_users)

                found_churn_users = total_churn[total_churn['probability'] >= cutoff]
                num_found_churn = len(found_churn_users)

                recall_churn = self._recall(num_found_churn, num_total_churn)
                precision_churn = self._precision(num_correct_churn, num_predicted_churn)
                if _math.isnan(recall_churn) or _math.isnan(precision_churn):
                    continue

                recall.append(recall_churn)
                precision.append(precision_churn)
                predicted_churn.append(num_predicted_churn)
                correct_prediction.append(num_correct_churn)
                found_churn.append(num_found_churn)
                cutoff_collection.append(cutoff)

            ret[model_name]['precision_recall'] = _gl.SFrame({
                'recall': recall,
                'precision': precision,
                'predicted_churn': predicted_churn,
                'correct_prediction': correct_prediction,
                'found_churn': found_churn,
                'cutoff': cutoff_collection
            })
        return ret

    def get_distribution(self):
        '''
        Get a dictionary with the fixed bin histogram of the churn probabilities
        for the churn_predictor and the baseline model. Each model returns an
        SFrame with a 10 rows (one for each bin) and 11 columns:

            bin_id               : An id for this bin.
            count                : Number of predictions in this bin.
            fraction             : Number of users as a fraction.
            mean_probability     : Average prediction probability in this bin.
            churn_count          : Number of churn predictions in this bin.
            churn_fraction       : Number of churned predictions as a fraction.
            bin_start            : Start of the bin (fraction of users)
            bin_end              : End of the bin (fraction of users)

        Example output
        --------------
        {
            'baseline': SFrame(
                +--------+-------+-------------+------------------+------------------+
                | bin_id | count | churn_count |   fraction       | mean_probability |
                +--------+-------+-------------+------------------+------------------+
                |   0    |  350  |      12     |  0.37674919268   |  0.0542312       |
                |   1    |   38  |      4      | 0.0409041980624  |  0.1723132       |
                |   2    |   10  |      3      | 0.010764262648   |  0.2498258       |
                |   3    |   4   |      1      | 0.0043057050592  |  0.3767491       |
                |   4    |   3   |      2      | 0.0032292787944  |  0.42985129      |
                |   5    |   4   |      2      | 0.0043057050592  |  0.5578231       |
                |   6    |   9   |      7      | 0.00968783638321 |  0.671238184     |
                |   7    |   6   |      5      | 0.00645855758881 |  0.74089238      |
                |   8    |   11  |      11     | 0.0118406889128  |  0.85098238      |
                |   9    |  494  |     493     | 0.531754574812   |  0.96038230      |
                +--------+-------+-------------+------------------+------------------+
                +------------------+-----------+---------+
                | churn_fraction   | bin_start | bin_end |
                +------------------+-----------+---------+
                |  0.0129171151776 |    0.0    |   0.1   |
                |  0.0043057050592 |    0.1    |   0.2   |
                |  0.0032292787944 |    0.2    |   0.3   |
                |  0.0010764262648 |    0.3    |   0.4   |
                |  0.0021528525296 |    0.4    |   0.5   |
                |  0.0021528525296 |    0.5    |   0.6   |
                |  0.00753498385361|    0.6    |   0.7   |
                |  0.005382131324  |    0.7    |   0.8   |
                |  0.0118406889128 |    0.8    |   0.9   |
                |  0.530678148547  |    0.9    |   1.0   |
                +------------------+-----------+---------+
            ),
            'churn_predictor': SFrame(
                +--------+-------+-------------+------------------+------------------+
                | bin_id | count | churn_count |   fraction       | churn_fraction   |
                +--------+-------+-------------+------------------+------------------+
                |   0    |  350  |      12     |  0.37674919268   |  0.0129171151776 |
                |   1    |   38  |      4      | 0.0409041980624  |  0.0043057050592 |
                |   2    |   10  |      3      | 0.010764262648   |  0.0032292787944 |
                |   3    |   4   |      1      | 0.0043057050592  |  0.0010764262648 |
                |   4    |   3   |      2      | 0.0032292787944  |  0.0021528525296 |
                |   5    |   4   |      2      | 0.0043057050592  |  0.0021528525296 |
                |   6    |   9   |      7      | 0.00968783638321 |  0.00753498385361|
                |   7    |   6   |      5      | 0.00645855758881 |  0.005382131324  |
                |   8    |   11  |      11     | 0.0118406889128  |  0.0118406889128 |
                |   9    |  494  |     493     | 0.531754574812   |  0.530678148547  |
                +--------+-------+-------------+------------------+------------------+
                +-----------+---------+
                | bin_start | bin_end |
                +-----------+---------+
                |    0.0    |   0.1   |
                |    0.1    |   0.2   |
                |    0.2    |   0.3   |
                |    0.3    |   0.4   |
                |    0.4    |   0.5   |
                |    0.5    |   0.6   |
                |    0.6    |   0.7   |
                |    0.7    |   0.8   |
                |    0.8    |   0.9   |
                |    0.9    |   1.0   |
                +-----------+---------+
            )
        }
        '''
        return self._distribution
