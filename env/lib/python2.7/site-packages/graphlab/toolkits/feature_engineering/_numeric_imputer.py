import graphlab as _gl

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
            >>> sf = graphlab.SFrame({'a' : [1,2,3,4,5], 'b' : [2,3,4,2,3]})

            # Create the imputer for the features ['a', 'b'].
            >>> imputer = graphlab.feature_engineering.NumericImputer(
                            features = ['a', 'b'], strategy = 'mean')

            # Learn the sufficient stats required for imputation of each column.
            >>> imputer = imputer.fit(sf)

            # Return the list of mean for each of the columns.
            >>> imputer['means']

            Columns:
              a  float
              b  float

            Rows: 1

            Data:
            +-----+-----+
            |  a  |  b  |
            +-----+-----+
            | 3.0 | 2.8 |
            +-----+-----+
            [1 rows x 2 columns]

'''

_fit_transform_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,None,4,5], 'b' : [2,3,None,5,6]})

            # Create the imputer for the features ['a', 'b'].
            >>> imputer = graphlab.feature_engineering.NumericImputer(
                                features = ['a', 'b'])

            # Fit and transform on the same data.
            >>> transformed_sf = imputer.fit_transform(sf)

            Columns:
                    a   float
                    b   float

            Rows: 5

            Data:
            +-----+-----+
            |  a  |  b  |
            +-----+-----+
            | 1.0 | 2.0 |
            | 2.0 | 3.0 |
            | 3.0 | 4.0 |
            | 4.0 | 5.0 |
            | 5.0 | 6.0 |
            +-----+-----+
            [5 rows x 2 columns]
'''

_transform_examples_doc = '''

            # Integer/Float columns
            # ----------------------------------------------------------------------
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,4,5], 'b' : [2,3,5,6]})

            # Create the imputer for the features ['a', 'b'].
            >>> imputer = graphlab.feature_engineering.NumericImputer(
                                features = ['a', 'b']).fit(sf)

            # Impute the missing values in new data.
            >>> sf_new = graphlab.SFrame({'a' : [1,2,None,4,5], 'b' : [2,3,None,5,6]})
            >>> transformed_sf = imputer.transform(sf_new)

            Columns:
                    a   float
                    b   float

            Rows: 5

            Data:
            +-----+-----+
            |  a  |  b  |
            +-----+-----+
            | 1.0 | 2.0 |
            | 2.0 | 3.0 |
            | 3.0 | 4.0 |
            | 4.0 | 5.0 |
            | 5.0 | 6.0 |
            +-----+-----+
            [5 rows x 2 columns]

            # Lists can contain numeric and None values.
            # ----------------------------------------------------------------------
            >>> sf = graphlab.SFrame({'lst': [[1, 2],
                                            [2, 3],
                                            [3, 4],
                                            [5, 6],
                                            [6, 7]]})

            # Construct and fit an imputer for the column['lst'].
            >>> from graphlab.toolkits.feature_engineering import NumericImputer
            >>> imputer = graphlab.feature_engineering.create(sf,
                            NumericImputer(features = ['lst']))

            # Impute the missing values in the new data.
            >>> new_sf = graphlab.SFrame({'lst': [[1, 2],
                                               [2, 3],
                                               [3, 4],
                                               [None, None],
                                               [5, 6],
                                               [6, 7]]})
            >>> transformed_sf = imputer.transform(sf)

            Columns:
                    a   list

            Rows: 6

            Data:
            +------------+
            |     lst    |
            +------------+
            |   [1, 2]   |
            |   [2, 3]   |
            |   [3, 4]   |
            | [3.4, 4.4] |
            |   [5, 6]   |
            |   [6, 7]   |
            +------------+
            [6 rows x 1 columns]

            # Dictionaries (Assumes sparse data format)
            # ----------------------------------------------------------------------

            # Construct and fit an imputer for the column ['dict'].
            >>> from graphlab.toolkits.feature_engineering import NumericImputer
            >>> sf = graphlab.SFrame({'dict':
                            [{'a':1, 'b': 2, 'c': 3},
                             {'a':0, 'b': 0, 'c': 0},
                             {'b':4, 'c': 0, 'd': 6}]})
            >>> imputer = graphlab.toolkits.feature_engineering.create(sf,
                             NumericImputer(features = ['dict']))

            # Impute the missing values for the new data.
            >>> sf = graphlab.SFrame({'dict':
                            [{'a':1, 'b': 2, 'c': 3},
                             None,
                             {'b':4, 'c': None, 'd': 6}]})
            >>> transformed_sf = imputer.transform(sf)

            Columns:
              dict  dict

            Rows: 3

            Data:
            +-------------------------------+
            |              dict             |
            +-------------------------------+
            |    {'a': 1, 'c': 3, 'b': 2}   |
            | {'a': 0.3333333333333333, ... |
            |   {'c': 1.0, 'b': 4, 'd': 6}  |
            +-------------------------------+
            [3 rows x 1 columns]

'''

@republish_docs
class NumericImputer(Transformer):
    '''
    Impute missing values with feature means.

    Input columns to the NumericImputer must be of type *int*, *float*,
    *dict*, *list*, or *array.array*.  For each column in the input, the transformed output is
    a column where the input is retained as is if:

     * there is no missing value.

    Inputs that do not satisfy the above are set to the mean value of that
    feature.

    The behavior for different input data column types is as follows:
    (see :func:`~graphlab.feature_engineering.NumericImputer.transform` for
    for examples).


    * **float** : If there is a missing value, it is replaced with the mean
      of that column.

    * **int**   : Behaves the same way as *float*.

    * **list**  : Each index of the list is treated as a feature column, and
      missing values are replaced with per-feature means. This is the same as
      unpacking, computing the mean, and re-packing. All elements must be of
      type *float*, *int*, or *None*. See :func:`~graphlab.SFrame.pack_columns`
      for more information.

    * **array** : Same behavior as *list*

    * **dict**  : Same behavior as *list*, except keys not present in a
      particular row are implicitly interpreted as having the value 0. This
      makes the  *dict* type a sparse representation of a vector.


    Parameters
    ----------
    features : list[str] | str | None, optional
        Name(s) of feature column(s) to be transformed. If set to None, then all
        feature columns are used.

    excluded_features : list[str] | str | None, optional
        Name(s) of feature columns in the input dataset to be ignored. Either
        `excluded_features` or `features` can be passed, but not both.

    strategy: 'auto'|'mean', optional
        The strategy with which to perform imputation.Currently can be 'auto'
        or 'mean'. Both currently perform mean imputation.

    output_column_prefix : str, optional
        The prefix to use for the column name of each transformed column.
        When provided, the transformation will add columns to the input data,
        where the new name is "`output_column_prefix`.original_column_name".
        If `output_column_prefix=None` (default), then the output column name
        is the same as the original feature column name.

    Returns
    -------
    out : NumericImputer
        A NumericImputer object which is initialized with the defined parameters.

    See Also
    --------
    graphlab.toolkits.feature_engineering._numeric_imputer.NumericImputer
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

        # Create data.
        >>> sf = graphlab.SFrame({'a': [1,3], 'b' : [2,4]})

        # Create a transformer.
        >>> from graphlab.toolkits.feature_engineering import NumericImputer
        >>> imputer = graphlab.feature_engineering.create(sf,
                     NumericImputer(features = ['a', 'b'], strategy = 'mean'))

        # Transform the data.
        >>> new_sf = graphlab.SFrame({'a': [1,None,3], 'b' : [2, None,4]})
        >>> transformed_sf = imputer.transform(new_sf)

        # Save the transformer.
        >>> imputer.save('save-path')

        # Return the means.
        >>> imputer['means']
        Columns:
            a  float
            b  float

        Rows: 1

        Data:
        +-----+-----+
        |  a  |  b  |
        +-----+-----+
        | 2.0 | 3.0 |
        +-----+-----+
        [1 rows x 2 columns]
    '''
    # Doc strings
    _fit_examples_doc = _fit_examples_doc
    _transform_examples_doc = _transform_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc

    # Default options
    get_default_options = staticmethod(_get_default_options_wrapper(
            '_MeanImputer', 'toolkits.feature_engineering._mean_imputer',
                                                'MeanImputer', True))

    def __init__(self, features=None, excluded_features=None, strategy='auto',
                 output_column_prefix=None):

        # Process and make a copy of the features, exclude.
        _features, _exclude = _internal_utils.process_features(
                                        features, excluded_features)

        # Type checking
        _raise_error_if_not_of_type(strategy, [str])

        # Set up options
        opts = {
                'strategy' : strategy,
                'output_column_prefix': output_column_prefix
                }
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features

        # Initialize object
        proxy = _gl.extensions._MeanImputer()
        proxy.init_transformer(opts)
        super(NumericImputer, self).__init__(proxy, self.__class__)

    def _get_summary_struct(self):
        _features = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('features')))
        _exclude = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('excluded_features')))
        fields = [
            ("Features", _features),
            ("Excluded features", _exclude),
        ]
        section_titles = ['Model fields']
        return ([fields], section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, 30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        imputer = _gl.feature_engineering.NumericImputer(
                            features = ['a', 'b'], strategy = 'mean')
        return imputer.fit(sf), sf
