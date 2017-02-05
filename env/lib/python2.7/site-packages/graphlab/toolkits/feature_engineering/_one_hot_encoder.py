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

            # Use the top-2 most frequent categories per column for features ['a', 'b'].
            >>> encoder = graphlab.feature_engineering.OneHotEncoder(
                    features = ['a', 'b'], max_categories = 2)

            # Fit i.e learn the mapping for each column.
            >>> encoder = encoder.fit(sf)

            # Return the indices in the encoding.
            >>> encoder['feature_encoding']

            Columns:
                      feature    str
                      category    str
                      index    int

            Rows: 4

            Data:
            +---------+----------+-------+
            | feature | category | index |
            +---------+----------+-------+
            |    a    |    2     |   0   |
            |    a    |    3     |   1   |
            |    b    |    2     |   2   |
            |    b    |    3     |   3   |
            +---------+----------+-------+
            [4 rows x 3 columns]
'''

_fit_transform_examples_doc = '''
            # Create the data
            >>> from graphlab.toolkits.feature_engineering import OneHotEncoder
            >>> sf = graphlab.SFrame({'a' : [1,2,3,2,3], 'b' : [2,3,4,2,3]})

            # Create a one-hot encoder for the features ['a', 'b'].
            >>> encoder = graphlab.feature_engineering.create(sf,
                            OneHotEncoder(features = ['a', 'b']))

            # Transform the data.
            >>> transformed_sf = encoder.transform(sf)
            Columns:
                feature    str
                category   str
                index      int

            Rows: 4

            Data:
            +---------+----------+-------+
            | feature | category | index |
            +---------+----------+-------+
            |    a    |    2     |   0   |
            |    a    |    3     |   1   |
            |    b    |    2     |   2   |
            |    b    |    3     |   3   |
            +---------+----------+-------+
            [4 rows x 3 columns]

            # Use the top-2 most frequent categories per column for features ['a', 'b'].
            >>> encoder = graphlab.feature_engineering.OneHotEncoder(
                    features = ['a', 'b'], max_categories = 2)

            # Fit and transform on the same data.
            >>> transformed_sf = encoder.fit_transform(sf)
            Columns:
                encoded_features    dict

            Rows: 5

            Data:
            +------------------+
            | encoded_features |
            +------------------+
            |      {2: 1}      |
            |   {0: 1, 3: 1}   |
            |      {1: 1}      |
            |   {0: 1, 2: 1}   |
            |   {1: 1, 3: 1}   |
            +------------------+
            [5 rows x 1 columns]
'''

_transform_examples_doc = '''
            # String/Integer columns
            # ----------------------------------------------------------------------
            >>> from graphlab.toolkits.feature_engineering import OneHotEncoder
            >>> sf = graphlab.SFrame({'a' : [1,2,3,2,3], 'b' : [2,3,4,2,3]})

            # Create a OneHotEncoder for features ['a', 'b']
            >>> encoder = graphlab.feature_engineering.create(sf,
                    OneHotEncoder(features = ['a', 'b']))

            # Fit and transform on the same data.
            >>> transformed_sf = encoder.fit_transform(sf)
            Columns:
                encoded_features    dict

            Rows: 5

            Data:
            +--------------------+
            |  encoded_features  |
            +--------------------+
            | {0: 1, 1: 1, 2: 1} |
            | {2: 1, 3: 1, 4: 1} |
            | {5: 1, 6: 1, 7: 1} |
            | {2: 1, 3: 1, 4: 1} |
            | {0: 1, 3: 1, 6: 1} |
            +--------------------+
            [5 rows x 1 columns]

            # Lists can be used to encode sets of categories for each example.
            # ----------------------------------------------------------------------
            >>> from graphlab.toolkits.feature_engineering import OneHotEncoder
            >>> sf = graphlab.SFrame({'categories': [['cat', 'mammal'],
                                                     ['dog', 'mammal'],
                                                     ['human', 'mammal'],
                                                     ['seahawk', 'bird'],
                                                     ['wasp', 'insect']]})

            # Construct and fit.
            >>> encoder = graphlab.feature_engineering.create(sf,
                        OneHotEncoder(features = ['categories']))

            # Transform the data
            >>> transformed_sf = encoder.transform(sf)
            Columns:
                encoded_features    dict

            Rows: 5

            Data:
            +------------------+
            | encoded_features |
            +------------------+
            |   {0: 1, 1: 1}   |
            |   {0: 1, 2: 1}   |
            |   {0: 1, 3: 1}   |
            |   {4: 1, 5: 1}   |
            |   {6: 1, 7: 1}   |
            +------------------+
            [5 rows x 1 columns]

            # Dictionaries can be used for name spaces & sub-categories.
            # ----------------------------------------------------------------------
            >>> from graphlab.toolkits.feature_engineering import OneHotEncoder
            >>> sf = graphlab.SFrame({'attributes':
                            [{'height':'tall', 'age': 'senior', 'weight': 'thin'},
                             {'height':'short', 'age': 'child', 'weight': 'thin'},
                             {'height':'giant', 'age': 'adult', 'weight': 'fat'},
                             {'height':'short', 'age': 'child', 'weight': 'thin'},
                             {'height':'tall', 'age': 'child', 'weight': 'fat'}]})

            # Construct and fit.
            >>> encoder = graphlab.feature_engineering.create(sf,
                                    OneHotEncoder(features = ['attributes']))

            # Transform the data
            >>> transformed_sf = encoder.transform(sf)
            Columns:
                encoded_features    dict

            Rows: 5

            Data:
            +--------------------+
            |  encoded_features  |
            +--------------------+
            | {0: 1, 1: 1, 2: 1} |
            | {2: 1, 3: 1, 4: 1} |
            | {5: 1, 6: 1, 7: 1} |
            | {2: 1, 3: 1, 4: 1} |
            | {0: 1, 3: 1, 6: 1} |
            +--------------------+
            [5 rows x 1 columns]
'''

@republish_docs
class OneHotEncoder(Transformer):
    '''
    Encode a collection of categorical features using a *1-of-K* encoding scheme.

    Input columns to the one-hot-encoder must by of type *int*, *string*,
    *dict*, or *list*. The transformed output is a column of type dictionary
    (`max_categories` per column dimension sparse vector) where the key
    corresponds to the index of the categorical variable and the value is `1`.

    The behaviour of the one-hot-encoder for each input data column type is as
    follows. (see :func:`~graphlab.feature_engineering.OneHotEncoder.transform`
    for examples of the same).


    * **string** : The key in the output dictionary is the string category and
      the value is 1.

    * **int** : Behave similar to *string* columns.

    * **list** : Each value in the list is treated like an individual string.
      Hence, a *list* of categorical variables can be used to represent a
      feature where all categories in the list are simultaneously `hot`.

    * **dict** : They key of the dictionary is treated as a `namespace` and the
      value is treated as a `sub-category` in the `namespace`. The categorical
      variable being encoded in this case is a combination of the `namespace`
      and the `sub-category`.


    Parameters
    ----------
    features : list[str] | str | None, optional
        Name(s) of feature column(s) to be transformed. If set to None, then all
        columns are used.

    excluded_features : list[str] | str | None, optional
        Name(s) of feature columns in the input dataset to be ignored. Either
        `excluded_features` or `features` can be passed, but not both.

    max_categories: int, optional
        The maximum number of categories (per feature column) to use in the
        encoding. If the number of unique categories in a column exceed
        `max_categories`, then only the most frequent used categories are retained.
        If set to None, then all categories in the column are used.

    output_column_name : str, optional
        The name of the output column. If the column already exists, then a
        suffix is appended to the name.

    Returns
    -------
    out : OneHotEncoder
        A OneHotEncoder object which is initialized with the defined
        parameters.

    Notes
    -------
    - `None` values are treated as separate categories and are encoded along with the rest of the values.

    See Also
    --------
    graphlab.toolkits.feature_engineering._count_thresholder.OneHotEnconder, graphlab.toolkits.feature_engineering.create

    Examples
    --------

    .. sourcecode:: python

        # Create data.
        >>> sf = graphlab.SFrame({'a': [1,2,3], 'b' : [2,3,4]})

        # Create a one-hot encoder on the features ['a', 'b'].
        >>> from graphlab.toolkits.feature_engineering import OneHotEncoder
        >>> encoder = graphlab.feature_engineering.create(sf,
                            OneHotEncoder(features = ['a', 'b']))

        # Transform data.
        >>> transformed_sf = encoder.transform(sf)
        Columns:
        encoded_features        dict

        Rows: 3

        Data:
        +------------------+
        | encoded_features |
        +------------------+
        |   {0: 1, 3: 1}   |
        |   {1: 1, 4: 1}   |
        |   {2: 1, 5: 1}   |
        +------------------+
        [3 rows x 1 columns]

        # Save the transformer.
        >>> encoder.save('save-path')

        # Return the indices in the encoding.
        >>> encoder['feature_encoding']
        Columns:
                feature str
                category        str
                index   int

        Rows: 6

        Data:
        +---------+----------+-------+
        | feature | category | index |
        +---------+----------+-------+
        |    a    |    1     |   0   |
        |    a    |    2     |   1   |
        |    a    |    3     |   2   |
        |    b    |    2     |   3   |
        |    b    |    3     |   4   |
        |    b    |    4     |   5   |
        +---------+----------+-------+
    '''

    # Doc strings
    _fit_examples_doc = _fit_examples_doc
    _transform_examples_doc = _transform_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc

    # Default options
    get_default_options = staticmethod(_get_default_options_wrapper(
            '_OneHotEncoder', 'toolkits.feature_engineering._one_hot_encoder',
                                                'OneHotEncoder', True))

    def __init__(self, features=None, excluded_features=None,
            max_categories=None, output_column_name = 'encoded_features'):

        # Process and make a copy of the features, exclude.
        _features, _exclude = _internal_utils.process_features(
                                        features, excluded_features)

        # Type checking
        _raise_error_if_not_of_type(max_categories, [int, type(None)])
        _raise_error_if_not_of_type(output_column_name, [str])

        # Set up options
        opts = {
          'max_categories': max_categories,
          'output_column_name': output_column_name,
        }
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features

        # Initialize object
        proxy = _gl.extensions._OneHotEncoder()
        proxy.init_transformer(opts)
        super(OneHotEncoder, self).__init__(proxy, self.__class__)

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
            ("Output column name", 'output_column_name'),
            ("Max categories per column", 'max_categories'),
        ]
        section_titles = ['Model fields']

        return ([fields], section_titles)

    def __repr__(self):
        """
        Return a string description of the model, including a description of
        the training data, training statistics, and model hyper-parameters.

        Returns
        -------
        out : string
            A description of the model.
        """
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a' : [1, 2, 3, 2, 3], 'b' : [2, 3, 4, 2, 3]})
        encoder = _gl.feature_engineering.OneHotEncoder(
                    features = ['a', 'b'], max_categories = 2)
        return encoder.fit(sf), sf


