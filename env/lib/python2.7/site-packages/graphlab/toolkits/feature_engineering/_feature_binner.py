import graphlab as _gl
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits.feature_engineering._feature_engineering import Transformer
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _precomputed_field
from graphlab.util import _raise_error_if_not_of_type

from . import _internal_utils
from ._doc_utils import republish_docs

_fit_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,3], 'b' : [2,3,4]})

            # Bin the features ['a', b']
            >>> binner = graphlab.feature_engineering.FeatureBinner(
                                features = ['a', 'b'], strategy='quantile')
            >>> fit_binner = binner.fit(sf)

            # Describe the bins in detail
            >>> fit_binner['bins']
            +--------+------+---------------------+--------------------+
            | column | name |         left        |       right        |
            +--------+------+---------------------+--------------------+
            |   a    | a_0  | -1.79769313486e+308 |        1.0         |
            |   a    | a_1  |         1.0         |        1.0         |
            |   a    | a_2  |         1.0         |        1.0         |
            |   a    | a_3  |         1.0         |        1.0         |
            |   a    | a_4  |         1.0         |        2.0         |
            |   a    | a_5  |         2.0         |        2.0         |
            |   a    | a_6  |         2.0         |        2.0         |
            |   a    | a_7  |         2.0         |        3.0         |
            |   a    | a_8  |         3.0         |        3.0         |
            |   a    | a_9  |         3.0         | 1.79769313486e+308 |
            +--------+------+---------------------+--------------------+
            [20 rows x 4 columns]
            Note: Only the head of the SFrame is printed.
            You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
'''

_fit_transform_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,3], 'b' : [2,3,4]})

            # Fit and transform on the data.
            >>> binner = graphlab.feature_engineering.FeatureBinner(
                            features = ['a', 'b'], strategy = 'logarithmic')
            >>> binned_sf = binner.fit_transform(sf)
            +-----+-----+
            |  a  |  b  |
            +-----+-----+
            | a_0 | b_1 |
            | a_1 | b_1 |
            | a_1 | b_1 |
            +-----+-----+
            [3 rows x 2 columns]
'''

_transform_examples_doc = '''
            #  Logarithmic binning (default)
            # Create the data.
            >>> sf = graphlab.SFrame({'a' : range(100), 'b' : range(100)})

            # Fit the feature binner.
            >>> binner = graphlab.feature_engineering.FeatureBinner(
                            features = ['a', 'b'], strategy = 'logarithmic')
            >>> fit_binner = binner.fit(sf)

            # Transformed on the some new data).
            >>> new_sf = graphlab.SFrame({'a' : range(10), 'b' : range(10)})
            >>> binned_sf = fit_binner.transform(new_sf)
            +-----+-----+
            |  a  |  b  |
            +-----+-----+
            | a_0 | b_0 |
            | a_0 | b_0 |
            | a_1 | b_1 |
            | a_1 | b_1 |
            | a_1 | b_1 |
            | a_1 | b_1 |
            | a_1 | b_1 |
            | a_1 | b_1 |
            | a_1 | b_1 |
            | a_1 | b_1 |
            +-----+-----+
            [10 rows x 2 columns]

            #  Quantile binning
            # Create the data.
            >>> sf = graphlab.SFrame({'a' : range(100), 'b' : range(100)})

            # Fit the feature binner.
            >>> binner = graphlab.feature_engineering.FeatureBinner(
                            features = ['a', 'b'], strategy = 'quantile')
            >>> fit_binner = binner.fit(sf)

            # Transformed on the some new data).
            >>> new_sf = graphlab.SFrame({'a' : range(0, 100, 10), 'b' : range(0, 100, 10)})
            >>> binned_sf = fit_binner.transform(new_sf)
            +-----+-----+
            |  a  |  b  |
            +-----+-----+
            | a_0 | b_0 |
            | a_1 | b_1 |
            | a_2 | b_2 |
            | a_3 | b_3 |
            | a_4 | b_4 |
            | a_5 | b_5 |
            | a_6 | b_6 |
            | a_7 | b_7 |
            | a_8 | b_8 |
            | a_9 | b_9 |
            +-----+-----+
            [10 rows x 2 columns]
'''

@republish_docs
class FeatureBinner(Transformer):
    '''
    Feature binning is a method of turning continuous variables into categorical
    values.

    This is accomplished by grouping the values into a pre-defined number of bins.
    The continuous value then gets replaced by a string describing the bin
    that contains that value.

    FeatureBinner supports both 'logarithmic' and 'quantile' binning strategies
    for either int or float columns.

    Parameters
    ----------
    features : list[str] , optional
        Column names of features to be transformed. If None, all columns are
        selected.

    excluded_features : list[str] | str | None, optional
        Column names of features to be ignored in transformation. Can be string
        or list of strings. Either 'excluded_features' or 'features' can be
        passed, but not both.

    strategy : 'logarithmic' | 'quantiles', optional
        If the strategy is 'logarithmic', bin break points are defined by
        :math:`10^i` for i in [0,...,num_bins-2]. For instance, if
        num_bins = 2, the bins become (-Inf, 1], (1, Inf]. If num_bins = 3,
        the bins become (-Inf, 1], (1, 10], (10, Inf].

        If the strategy is 'quantile', the bin breaks are defined by the
        'num_bins'-quantiles for that columns data. Quantiles are values that
        separate the data into roughly equal-sized subsets.

    num_bins : int, optional
        The number of bins to group the continuous variables into.

    output_column_prefix : str, optional
        The prefix to use for the column name of each transformed column.
        When provided, the transformation will add columns to the input data,
        where the new name is "`output_column_prefix`.original_column_name".
        If `output_column_prefix=None` (default), then the output column name
        is the same as the original feature column name.

    Returns
    -------
    out : FeatureBinner
        A FeatureBinner object which is initialized with the defined
        parameters.

    See Also
    --------
    graphlab.toolkits.feature_engineering._feature_binner.FeatureBinner
    graphlab.toolkits.feature_engineering.create

    Notes
    -----
    - If the SFrame to be transformed already contains a column with the
      designated output column name, then that column will be replaced with the
      new output. In particular, this means that `output_column_prefix=None` will
      overwrite the original feature columns.

    Examples
    --------

    .. sourcecode:: python

        >>> from graphlab.toolkits.feature_engineering import *

        # Construct a feature binner with default options.
        >>> sf = graphlab.SFrame({'a': [1,2,3], 'b' : [2,3,4], 'c': [9,10,11]})
        >>> binner = graphlab.feature_engineering.create(sf,
              FeatureBinner(features = ['a', 'b', 'c'], strategy = 'quantile'))

        # Transform the data using the binner.
        >>> binned_sf = binner.transform(sf)

        # Save the transformer.
        >>> binner.save('save-path')

        # Return the details about the bins
        >>> binner['bins']

       Columns:
        column  str
        name    str
        left    float
        right   float

        Rows: 30

        Data:
        +--------+------+---------------------+--------------------+
        | column | name |         left        |       right        |
        +--------+------+---------------------+--------------------+
        |   a    | a_0  | -1.79769313486e+308 |        1.0         |
        |   a    | a_1  |         1.0         |        1.0         |
        |   a    | a_2  |         1.0         |        1.0         |
        |   a    | a_3  |         1.0         |        1.0         |
        |   a    | a_4  |         1.0         |        2.0         |
        |   a    | a_5  |         2.0         |        2.0         |
        |   a    | a_6  |         2.0         |        2.0         |
        |   a    | a_7  |         2.0         |        3.0         |
        |   a    | a_8  |         3.0         |        3.0         |
        |   a    | a_9  |         3.0         | 1.79769313486e+308 |
        +--------+------+---------------------+--------------------+
        [30 rows x 4 columns]
        Note: Only the head of the SFrame is printed.
        You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
        '''

    _fit_examples_doc = _fit_examples_doc
    _transform_examples_doc = _transform_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc

    get_default_options = staticmethod(_get_default_options_wrapper(
            '_FeatureBinner', 'toolkits.feature_engineering._feature_binner', 'FeatureBinner', True))

    def __init__(self, features=None, excluded_features=None,
                 strategy='logarithmic', num_bins=10,
                 output_column_prefix=None):

        # Process and make a copy of the features, exclude.
        _features, _exclude = _internal_utils.process_features(features, excluded_features)

        # Type checking
        _raise_error_if_not_of_type(num_bins, [int])
        _raise_error_if_not_of_type(strategy, [str])

        # Set up options
        opts = {
          'strategy': strategy,
          'num_bins': num_bins,
          'output_column_prefix': output_column_prefix
        }
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features

        # Initialize object
        proxy = _gl.extensions._FeatureBinner()
        proxy.init_transformer(opts)
        super(FeatureBinner, self).__init__(proxy, self.__class__)

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
            ("Features", _features),
            ("Excluded_features", _exclude),
            ("Strategy for creating bins", 'strategy'),
            ("Number of bins to use", 'num_bins')
        ]
        section_titles = ['Model fields']

        return ([fields], section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        binner = _gl.feature_engineering.FeatureBinner(
                                features = ['a', 'b'], strategy='quantile')
        return binner.fit(sf), sf
