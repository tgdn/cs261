import graphlab as _gl
import graphlab.connect as _mt
# Toolkit utils.
from graphlab.toolkits.feature_engineering._feature_engineering import Transformer
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _precomputed_field
from graphlab.util import _raise_error_if_not_of_type

# Feature engineering utils
from . import _internal_utils
from ._doc_utils import republish_docs

_fit_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,1,2], 'b' : [2,2,3], 'y':[0,1,0]})

            # Perform Count Featurization on columns 'a' and 'b' with respect to
            # the target 'y'
            >>> cf = graphlab.feature_engineering.CountFeaturizer(
                                features = ['a', 'b'], target = 'y')
            >>> cf = cf.fit(sf)
'''


_fit_transform_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,1,2], 'b' : [2,2,3], 'y':[0,1,0]})


            # Perform Count Featurization on columns 'a' and 'b' with respect to
            # the target 'y'
            >>> cf = graphlab.feature_engineering.CountFeaturizer(
                                    features = ['a', 'b'], target = 'y')

            # Generate the output features
            >>> out_sf = cf.fit_transform(sf)
            +------------+--------+------------+--------+---+
            |  count_a   | prob_a |  count_b   | prob_b | y |
            +------------+--------+------------+--------+---+
            | [1.0, 1.0] | [0.5]  | [1.0, 1.0] | [0.5]  | 0 |
            | [1.0, 1.0] | [0.5]  | [1.0, 1.0] | [0.5]  | 1 |
            | [1.0, 0.0] | [1.0]  | [1.0, 0.0] | [1.0]  | 0 |
            +------------+--------+------------+--------+---+
            [3 rows x 5 columns]
'''

_transform_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,1,2], 'b' : [2,2,3], 'y':[0,1,0]})

            # Perform Count Featurization on columns 'a' and 'b' with respect to
            # the target 'y'
            >>> cf = graphlab.feature_engineering.CountFeaturizer(
                                    features = ['a', 'b'], target = 'y')
            >>> cf = cf.fit(sf)

            # Generate the output features
            >>> out_sf = cf.transform(sf)
            +------------+--------+------------+--------+---+
            |  count_a   | prob_a |  count_b   | prob_b | y |
            +------------+--------+------------+--------+---+
            | [1.0, 1.0] | [0.5]  | [1.0, 1.0] | [0.5]  | 0 |
            | [1.0, 1.0] | [0.5]  | [1.0, 1.0] | [0.5]  | 1 |
            | [1.0, 0.0] | [1.0]  | [1.0, 0.0] | [1.0]  | 0 |
            +------------+--------+------------+--------+---+
            [3 rows x 5 columns]

            # Create the data
            >>> sf = graphlab.SFrame({'a' : ['a','a','b'], 'b' : ['b','b','c'], 'y':['0','1','0']})

            # Perform Count Featurization on columns 'a' and 'b' with respect to
            # the target 'y'
            >>> cf = graphlab.feature_engineering.CountFeaturizer(
                                    features = ['a', 'b'], target = 'y')
            >>> cf = cf.fit(sf)
            >>> out_sf = cf.transform(sf)
            +------------+--------+------------+--------+---+
            |  count_a   | prob_a |  count_b   | prob_b | y |
            +------------+--------+------------+--------+---+
            | [1.0, 1.0] | [0.5]  | [1.0, 1.0] | [0.5]  | 0 |
            | [1.0, 1.0] | [0.5]  | [1.0, 1.0] | [0.5]  | 1 |
            | [1.0, 0.0] | [1.0]  | [1.0, 0.0] | [1.0]  | 0 |
            +------------+--------+------------+--------+---+
            [3 rows x 5 columns]
'''

@republish_docs
class CountFeaturizer(Transformer):
    '''
    Replaces a collection of categorical columns with counts of a target column.

    The CountFeaturizer is an efficient way of reducing high dimensional
    categorical columns into simple counts for the purpose of classification.
    Supported types are only str and int and both are interpreted
    categorically. The CountFeaturizer is effective for significantly
    accelerating downstream learning procedures without loss of accuracy for
    extremely large datasets.

    Assuming we are going to try to predict column Y which has K unique classses.
    Then for every column X, we replace it with 2 columns,
    "count_X" and "prob_X". The column count_X contains an array of length K
    which contains the counts of each unique value of Y where X is fixed. The
    column prob_X contains the normalized value of count_X dropping the last
    value.

    For instance, given the following SFrame:

    .. sourcecode:: python

        >>> sf = graphlab.SFrame({'a' : [1,1,2], 'y':[0,1,0]})
        +---+---+
        | a | y |
        +---+---+
        | 1 | 0 |
        | 1 | 1 |
        | 2 | 0 |
        +---+---+

    After fit_transform the output SFrame is

    .. sourcecode:: python

        >>> cf = graphlab.feature_engineering.CountFeaturizer(target = 'y', laplace_smearing=0)
        >>> cf.fit_transform(sf)
        +------------+--------+---+
        |  count_a   | prob_a | y |
        +------------+--------+---+
        | [1.0, 1.0] | [0.5]  | 0 |
        | [1.0, 1.0] | [0.5]  | 1 |
        | [1.0, 0.0] | [1.0]  | 0 |
        +------------+--------+---+
        [3 rows x 3 columns]

    Observe that in the original sframe, there is 1 occurance where a = 1 & y =
    0 and 1 occurance where a = 1 & y = 1. Thus in every row where a = 1, we
    output [1.0, 1.0] in the count_a column.  Similarly, for the case of a = 2,
    we have a count of 1 where y = 0 & a = 2, and no occurances of y = 1 & a =
    2. Hence in every row where a = 2, we output [1.0, 0.0] in the count_a
    column. The prob_a column is just count_a column, normalized to sum to 1,
    and dropping the last value.

    The laplace_smearing parameter controls the amount of noise added to the
    result which may will allow fit() and transform() to be performed on the
    same dataset.  Tuning this parameter can be difficult in practice however.
    Therefore it is highly recommended (and is the default behavior) to set
    laplace_smearing=0 and split the training dataset into two sets, where one
    set is used only in fit() and the other set used only in transform().

    Parameters
    ----------
    target: str, required
        The target column we are trying to predict.

    features : list[str] | str | None, optional
        Name(s) of feature column(s) to be transformed. If set to None, then all
        columns are used.

    excluded_features : list[str] | str | None, optional
        Name(s) of feature columns in the input dataset to be ignored. Either
        `excluded_features` or `features` can be passed, but not both.

    num_bits : int, optional
        This parameter is the size of the countmin sketch used to approximate
        the counts and controls the accuracy of the counts.  The higher the
        value, the more accurate the counts, but takes up more memory. Defaults
        to 20.

    laplace_smearing : float, optional
        Defaults to 0. Adds some noise to the transform result to allow the same
        dataset to be used for both fit and transform. When the number of rows
        is small, this parameter can be reduced in value. If set to 0, it is
        recommended that the training set be split into two sets, where
        one set is used used in fit(), the the other used in transform().

    random_seed : int, optional
        A random seed. Fix this to get deterministic outcomes.

    count_column_prefix : str, optional
        The prefix added to the input column name to produce the output column
        name containing the counts. Defaults to `count_`

    prob_column_prefix : str, optional
        The prefix added to the input column name to produce the output column
        name containing the normalized counts. Defaults to `prob_`

    Returns
    -------
    out : CountFeaturizer
        A CountFeaturizer object which is initialized with the defined
        parameters.

    Notes
    -----
    The prob_X columns have one value dropped to eliminate a linear dependency.

    References
    ----------
    Implements the method described in `this blog
    <https://blogs.technet.microsoft.com/machinelearning/2015/02/17/big-learning-made-easy-with-counts/>`.


    Examples
    --------

    .. sourcecode:: python

        >>> from graphlab.toolkits.feature_engineering import *

        # Perform Count Featurization on columns 'a' and 'b' with respect to
        # the target 'y'
        >>> sf = graphlab.SFrame({'a' : [1,1,2], 'b' : [2,2,3], 'y':[0,1,0]})
        >>> cf = graphlab.feature_engineering.create(sf,
        ...               graphlab.feature_engineering.CountFeaturizer(
        ...                     features = ['a', 'b'], target = 'y'))

        # Transform the data
        >>> out_sf = cf.fit_transform(sf)
        >>> out_sf
        +------------+--------+------------+--------+---+
        |  count_a   | prob_a |  count_b   | prob_b | y |
        +------------+--------+------------+--------+---+
        | [1.0, 1.0] | [0.5]  | [1.0, 1.0] | [0.5]  | 0 |
        | [1.0, 1.0] | [0.5]  | [1.0, 1.0] | [0.5]  | 1 |
        | [1.0, 0.0] | [1.0]  | [1.0, 0.0] | [1.0]  | 0 |
        +------------+--------+------------+--------+---+
        [3 rows x 5 columns]

        # Save the transformer.
        >>> cf.save('save-path')
    '''

    _fit_examples_doc = _fit_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc
    _transform_examples_doc = _transform_examples_doc


    get_default_options = staticmethod(_get_default_options_wrapper(
        '_CountFeaturizer', 'toolkits.feature_engineering._count_featurizer',
                                                    'CountFeaturizer', True))

    _metric_handle = 'toolkits.feature_engineering.count_featurizer'

    def __init__(self, target, features=None, excluded_features=None,
            random_seed=None, laplace_smearing=0.0, num_bits=20,
            count_column_prefix='count_', prob_column_prefix='prob_'):

        _mt._get_metric_tracker().track(self._metric_handle + '.__init__')

        if count_column_prefix == prob_column_prefix:
            raise RuntimeError("count_column_prefix cannot be equal to prob_column_prefix")

        # Process and make a copy of the features, exclude.
        _features, _exclude = _internal_utils.process_features(
                                        features, excluded_features)

        # Type checking
        _raise_error_if_not_of_type(num_bits, [int])

        # Set up options
        opts = {
            'target':target,
            'num_bits': num_bits,
            'random_seed': random_seed,
            'laplace_smearing':laplace_smearing,
            'num_bits':num_bits,
            'count_column_prefix':count_column_prefix,
            'prob_column_prefix':prob_column_prefix
            }
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features

        # Initialize object
        proxy = _gl.extensions._CountFeaturizer()
        proxy.init_transformer(opts)
        super(CountFeaturizer, self).__init__(proxy, self.__class__)

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
        _features = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('features')))
        _exclude = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('excluded_features')))
        fields = [
            ("Target", "target"),
            ("Features", _features),
            ("Excluded features", _exclude),
            ("Number of bits", 'num_bits'),
            ("Random seed", 'random_seed'),
            ("Laplace Smearing", 'laplace_smearing'),
            ("Count Column Prefix", 'count_column_prefix'),
            ("Probability Column Prefix", 'prob_column_prefix')
            ]
        section_titles = [ 'Model fields' ]
        return ([fields], section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width= 30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a' : [1,1,2], 'b' : [2,2,3], 'y':[0,1,0]})
        cf = _gl.feature_engineering.CountFeaturizer(features = ['a', 'b'], target='y')
        return cf.fit(sf), sf
