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
            >>> sf = graphlab.SFrame({'a' : [1,2,3,2,3], 'b' : [2,3,4,2,3]})

            # Set all categories that did not occur at least 2 times to 'junk'.
            >>> count_tr = graphlab.feature_engineering.CountThresholder(
                    features = ['a', 'b'], threshold = 2, output_category_name = 'junk')

            # Fit i.e learn the mapping for each column.
            >>> count_tr = count_tr.fit(sf)

            # Return the list of categories for each of the columns.
            >>> count_tr['categories']

            Columns:
               feature  str
               category  str

            Rows: 4

            Data:
            +---------+----------+
            | feature | category |
            +---------+----------+
            |    a    |    2     |
            |    a    |    3     |
            |    b    |    2     |
            |    b    |    3     |
            +---------+----------+
            [4 rows x 2 columns]

'''

_fit_transform_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,3,2,3], 'b' : [2,3,4,2,3]})

            # Set all categories that did not occur at least 2 times to None.
            >>> count_tr = graphlab.feature_engineering.CountThresholder(
                     features = ['a', 'b'], threshold = 2, output_category_name = None)

            # Fit and transform on the same data.
            >>> transformed_sf = count_tr.fit_transform(sf)

            Columns:
            a  dict
            b  dict

            Rows: 3

            Data:
            +-------+--------+
            |   a   |   b    |
            +-------+--------+
            | None  |    2   |
            |   2   |    3   |
            |   3   |  None  |
            |   2   |    2   |
            |   3   |    3   |
            +-------+--------+
            [5 rows x 2 columns]
'''

_transform_examples_doc = '''
            # String/Integer columns
            # ----------------------------------------------------------------------
            >>> sf = graphlab.SFrame({'a' : [1,2,3,2,3], 'b' : [2,3,4,2,3]})

            # Set all categories that did not occur at least 2 times to None.
            >>> count_tr = graphlab.feature_engineering.CountThresholder(
                    features = ['a', 'b'], threshold = 2)

            # Fit and transform on the same data.
            >>> transformed_sf = count_tr.fit_transform(sf)

            Columns:
            a  dict
            b  dict

            Rows: 3

            Data:
            +-------+--------+
            |   a   |   b    |
            +-------+--------+
            | None  |    2   |
            |   2   |    3   |
            |   3   |  None  |
            |   2   |    2   |
            |   3   |    3   |
            +-------+--------+
            [5 rows x 2 columns]

            # Lists can be used to encode sets of categories for each example.
            # ----------------------------------------------------------------------
            >>> sf = graphlab.SFrame({'categories': [['cat', 'mammal'],
                                                     ['cat', 'mammal'],
                                                     ['human', 'mammal'],
                                                     ['seahawk', 'bird'],
                                                     ['duck', 'bird'],
                                                     ['seahawk', 'bird']]})

            # Construct and fit.
            >>> from graphlab.feature_engineering import CountThresholder
            >>> count_tr = graphlab.feature_engineering.create(sf,
                   CountThresholder(features = ['categories'], threshold = 2))

            # Transform the data
            >>> transformed_sf = count_tr.transform(sf)
            Columns:
            categories  dict

            Rows: 5

            Data:
            +-----------------+
            |    categories   |
            +-----------------+
            |  [cat, mammal]  |
            |  [cat, mammal]  |
            |  [None, mammal] |
            | [seahawk, bird] |
            |   [None, bird]  |
            | [seahawk, bird] |
            |   [None, None]  |
            +-----------------+

            [5 rows x 1 columns]


            # Dictionaries can be used for name spaces & sub-categories.
            # ----------------------------------------------------------------------
            >>> sf = graphlab.SFrame({'attributes':
                            [{'height':'tall', 'age': 'senior', 'weight': 'thin'},
                             {'height':'short', 'age': 'child', 'weight': 'thin'},
                             {'height':'giant', 'age': 'adult', 'weight': 'fat'},
                             {'height':'short', 'age': 'child', 'weight': 'thin'},
                             {'height':'tall', 'age': 'child', 'weight': 'fat'}]})

            # Construct and fit.
            >>> from graphlab.feature_engineering import CountThresholder
            >>> count_tr = graphlab.feature_engineering.create(sf,
                        CountThresholder(features = ['attributes'], threshold = 2))

            # Transform the data
            >>> transformed_sf = count_tr.transform(sf)

            Columns:
                attributes  dict

            Rows: 5

            Data:
            +-------------------------------+
            |           attributes          |
            +-------------------------------+
            | {'age': None, 'weight': 't... |
            | {'age': 'child', 'weight':... |
            | {'age': None, 'weight': No... |
            | {'age': 'child', 'weight':... |
            | {'age': 'child', 'weight':... |
            +-------------------------------+
'''

@republish_docs
class CountThresholder(Transformer):
    '''
    Map infrequent categorical variables to a `new/separate` category.

    Input columns to the CountThresholder must be of type *int*, *string*,
    *dict*, or *list*.  For each column in the input, the transformed output is
    a column where the input category is retained as is if:

     * it has occurred at least `threshold` times in the training data.

    categories that does not satisfy the above are set to `output_category_name`.

    The behaviour for different input data column types is as follows:
    (see :func:`~graphlab.feature_engineering.CountThresholder.transform` for
    for examples).


    * **string** : Strings are marked with the `output_category_name` if the
      threshold condition described above is not satisfied.

    * **int** : Behave the same way as *string*. If `output_category_name` is
      of type *string*, then the entire column is cast to string.

    * **list** : Each of the values in the list are mapped in the same way as
      a string value.

    * **dict** : They key of the dictionary is treated as a `namespace` and the
      value is treated as a `sub-category` in the `namespace`. The categorical
      variable passed through the transformer is a combination of the
      `namespace` and the `sub-category`.


    Parameters
    ----------
    features : list[str] | str | None, optional
        Name(s) of feature column(s) to be transformed. If set to None, then all
        feature columns are used.

    excluded_features : list[str] | str | None, optional
        Name(s) of feature columns in the input dataset to be ignored. Either
        `excluded_features` or `features` can be passed, but not both.

    threshold : int, optional
        Ignore all categories that have not occurred at least `threshold` times.
        All categories that do not occur at least `threshold` times are
        mapped to the `output_category_name`.

    output_category_name : str | None, optional
        The value to use for the categories that do not satisfy the `threshold`
        condition.

    output_column_prefix : str, optional
        The prefix to use for the column name of each transformed column.
        When provided, the transformation will add columns to the input data,
        where the new name is "`output_column_prefix`.original_column_name".
        If `output_column_prefix=None` (default), then the output column name
        is the same as the original feature column name.

    Returns
    -------
    out : CountThresholder
        A CountThresholder object which is initialized with the defined parameters.

    See Also
    --------
    graphlab.toolkits.feature_engineering._count_thresholder.CountThresholder
    graphlab.toolkits.feature_engineering.create

    Notes
    -----
    - If the SFrame to be transformed already contains a column with the
      designated output column name, then that column will be replaced with the
      new output. In particular, this means that `output_column_prefix=None` will
      overwrite the original feature columns.
    - If the `output_category_name` and input feature column are not of the same
      type, then the output column is cast to `str`.
    - `None` values are treated as separate categories and are encoded along
      with the rest of the values.

    Examples
    --------

    .. sourcecode:: python

        # Create data.
        >>> sf = graphlab.SFrame({'a': [1,2,3], 'b' : [2,3,4]})

        # Create a transformer.
        >>> from graphlab.toolkits.feature_engineering import CountThresholder
        >>> count_tr = graphlab.feature_engineering.create(sf,
                CountThresholder(features = ['a', 'b'], threshold = 1))

        # Transform the data.
        >>> transformed_sf = count_tr.transform(sf)

        # Save the transformer.
        >>> count_tr.save('save-path')

        # Return the categories that are not discarded.
        >>> count_tr['categories']
        Columns:
                feature str
                category    str

        Rows: 6

        Data:
            +---------+----------+
            | feature | category |
            +---------+----------+
            |    a    |    1     |
            |    a    |    2     |
            |    a    |    3     |
            |    b    |    2     |
            |    b    |    3     |
            |    b    |    4     |
            +---------+----------+
            [6 rows x 2 columns]
    '''
    # Doc strings
    _fit_examples_doc = _fit_examples_doc
    _transform_examples_doc = _transform_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc

    # Default options
    get_default_options = staticmethod(_get_default_options_wrapper(
            '_CountThresholder', 'toolkits.feature_engineering._count_thresholder',
                                                'CountThresholder', True))

    def __init__(self, features=None, excluded_features=None, threshold=1,
                 output_category_name=None, output_column_prefix=None):

        # Process and make a copy of the features, exclude.
        _features, _exclude = _internal_utils.process_features(
                                        features, excluded_features)

        # Type checking
        _raise_error_if_not_of_type(threshold, [int, type(None)])

        # Set up options
        opts = {
          'threshold': threshold,
          'output_category_name': output_category_name,
          'output_column_prefix': output_column_prefix
        }
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features

        # Initialize object
        proxy = _gl.extensions._CountThresholder()
        proxy.init_transformer(opts)
        super(CountThresholder, self).__init__(proxy, self.__class__)

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
            ("Excluded features", _exclude),
            ("New category name", 'output_category_name'),
            ("Occurrence threshold", 'threshold'),
        ]
        section_titles = ['Model fields']

        return ([fields], section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, 30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a' : [1, 2, 3, 2, 3], 'b' : [2, 3, 4, 2, 3]})
        count_tr = _gl.feature_engineering.CountThresholder(
                    features = ['a', 'b'], threshold = 2,
                    output_category_name = 'junk')
        return count_tr.fit(sf), sf


