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
            >>> sf = graphlab.SFrame({'a' : [1,2,3], 'b' : [2,3,4]})

            # Hash the space of columns ['a', 'b'] into a single space.
            >>> hasher = graphlab.feature_engineering.FeatureHasher(
                                features = ['a', 'b'])
            >>> hasher = hasher.fit(sf)
'''


_fit_transform_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,3], 'b' : [2,3,4]})


            # Hash the space of columns ['a', 'b'] into a single space.
            >>> hasher = graphlab.feature_engineering.FeatureHasher(
                                    features = ['a', 'b'])

            # Hash and transform the space into a new space (called "hashed_features")
            >>> hashed_sf = hasher.fit_transform(sf)
            Columns:
                  hashed_features  dict

            Rows: 3

            Data:
            +-----------------+
            | hashed_features |
            +-----------------+
            | {937: 1, 59: 2} |
            | {937: 2, 59: 3} |
            | {937: 3, 59: 4} |
            +-----------------+
            [3 rows x 1 columns]
'''

_transform_examples_doc = '''

            # Numeric data.
            # -----------------------------------------------------------------

            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,3], 'b' : [2,3,4]})

            # Hash the space of columns ['a', 'b'] into a single space.
            >>> hasher = graphlab.feature_engineering.FeatureHasher(
                            features = ['a', 'b'])
            >>> hasher = hasher.fit(sf)

            # Transform the space into the hashed space (called "hashed_features")
            >>> hashed_sf = hasher.transform(sf)
            Columns:
                  hashed_features  dict

            Rows: 3

            Data:
            +-----------------+
            | hashed_features |
            +-----------------+
            | {937: 1, 59: 2} |
            | {937: 2, 59: 3} |
            | {937: 3, 59: 4} |
            +-----------------+
            [3 rows x 1 columns]

            # String/Categorical data
            # -----------------------------------------------------------------

            # Create the data
            >>> sf = graphlab.SFrame({'a' : ['a','b','c'], 'b' : ['d','e','f']})

            # Hash the feature space ['a', 'b']
            >>> hasher = graphlab.feature_engineering.FeatureHasher(
                            features = ['a', 'b'])
            >>> hasher = hasher.fit(sf)

            # Transform the data into the hashed space (called "hashed_features").
            >>> hashed_sf = hasher.transform(sf)

            Columns:
                    hashed_features dict

            Rows: 3

            Data:
            +------------------+
            | hashed_features  |
            +------------------+
            | {405: 1, 79: 1}  |
            | {454: 1, 423: 1} |
            | {308: 1, 36: 1}  |
            +------------------+
            [3 rows x 1 columns]

            # List/Vector data
            # -----------------------------------------------------------------
            # Create the data.
            >>> sf = graphlab.SFrame({'categories': [['cat', 'mammal'],
                                                     ['cat', 'mammal'],
                                                     ['human', 'mammal'],
                                                     ['seahawk', 'bird'],
                                                     ['duck', 'bird'],
                                                     ['seahawk', 'bird']]})


            # Hash the feature set
            >>> hasher = graphlab.feature_engineering.FeatureHasher(
                                        features = 'categories')
            >>> hasher = hasher.fit(sf)

            # Transform the data into the hashed space.
            >>> hashed_sf = hasher.transform(sf)

            Columns:
               hashed_features  dict

            Rows: 6

            Data:
            +-------------------------+
            |     hashed_features     |
            +-------------------------+
            | {111080: 1, 138612: -1} |
            | {111080: 1, 138612: -1} |
            |  {138612: -1, 21685: 1} |
            | {100073: 1, 178119: -1} |
            |  {100073: 1, 179522: 1} |
            | {100073: 1, 178119: -1} |
            +-------------------------+
            [6 rows x 1 columns]


            # Dictionary data
            # -----------------------------------------------------------------
            >>> sf = graphlab.SFrame({'attributes':
                            [{'height':'tall', 'age': 'senior', 'weight': 'thin'},
                             {'height':'short', 'age': 'child', 'weight': 'thin'},
                             {'height':'giant', 'age': 'adult', 'weight': 'fat'},
                             {'height':'short', 'age': 'child', 'weight': 'thin'},
                             {'height':'tall', 'age': 'child', 'weight': 'fat'}]})
            # Hash the feature set
            >>> hasher = graphlab.feature_engineering.FeatureHasher(
                                        features = 'attributes')
            >>> hasher = hasher.fit(sf)

            # Transform the data into the hashed space.
            >>> hashed_sf = hasher.transform(sf)

            Columns:
              hashed_features  dict

            Rows: 5

            Data:
            +-------------------------------+
            |        hashed_features        |
            +-------------------------------+
            | {136240: -1, 137879: 1, 12... |
            | {258151: -1, 199837: -1, 1... |
            | {26136: 1, 81585: -1, 1527... |
            | {258151: -1, 199837: -1, 1... |
            | {26136: 1, 258151: -1, 123... |
            +-------------------------------+
            [5 rows x 1 columns]
'''

@republish_docs
class FeatureHasher(Transformer):

    '''
    Hashes an input feature space to an n-bit feature space.

    Feature hashing is an efficient way of vectorizing features, and performing
    dimensionality reduction or expansion along the way. Supported types include
    array.array, list, dict, float, int, and string.  The behavior for creating
    keys and values for different input data column types is given below.

    * **array.array** : Keys are created by 1) combining the index
      of an element and the column name, 2) hashing the combination of the two.
      Each element in the array are the values in the returned dictionary.

    * **list** : Behaves the same as array.array, but if the element is non-numerical
      the element is combined with the column name and hashed, and 1 is used
      as the value.

    * **dict** : Each key in the dictionary is combined with the column name and hashed,
      and the value is kept. If the value is is non-numerical, the element is
      combined with the column name and hashed, and 1 is used as the value.

    * **float** : the column name is hashed, and the column
      entry becomes the value

    * **int** : Same behavior as float

    * **string** : Hash the string and use it as a key, and use 1 as the value.

    The hashed values are collapsed into a single sparse representation of a
    vector, so all hashed columns are replaced by a single column with name
    specified by 'output_column_name'.

    Parameters
    ----------
    features : list[str] | str | None, optional
        Name(s) of feature column(s) to be transformed. If set to None, then all
        columns are used.

    excluded_features : list[str] | str | None, optional
        Name(s) of feature columns in the input dataset to be ignored. Either
        `excluded_features` or `features` can be passed, but not both.

    num_bits : int, optional
        The number of bits to hash to. There will be :math:`2^{num\_bits}`
        indices in the resulting vector.

    output_column_name : str, optional
        The name of the output column. If the column already exists, then a
        suffix is append to the name.

    Returns
    -------
    out : FeatureHasher
        A FeatureHasher object which is initialized with the defined
        parameters.

    Notes
    -----
    - Each time a key is hashed, the corresponding value is multipled by
      either 1.0 or -1.0,  chosen with equal probability.  The final hashed
      feature value is the accumulation of values for all keys hashed to that
      bucket.

    References
    ----------
    - Collaborative Spam Filtering with the Hashing Trick. J. Attenberg,
      K. Q. Weinberger, A. Smola, A. Dasgupta, M. Zinkevich Virus Bulletin
      (VB) 2009.

    See Also
    --------
    graphlab.toolkits.feature_engineering._feature_hasher.FeatureHasher
    graphlab.toolkits.feature_engineering.create

    Examples
    --------

    .. sourcecode:: python

        from graphlab.toolkits.feature_engineering import *

        # Hash the feature space ['a', 'b, 'c'] into a single space.
        >>> sf = graphlab.SFrame({'a': [1,2,3], 'b' : [2,3,4], 'c': [9,10,11]})
        >>> hasher = graphlab.feature_engineering.create(sf,
                                FeatureHasher(features = ['a', 'b', 'c']))

        # Transform the data using the hasher.
        >>> hashed_sf = hasher.transform(sf)
        >>> hashed_sf

        Columns:
          hashed_features  dict

        Rows: 3

        Data:
        +-------------------------------+
        |        hashed_features        |
        +-------------------------------+
        | {79785: -1, 188475: -2, 21... |
        | {79785: -2, 188475: -3, 21... |
        | {79785: -3, 188475: -4, 21... |
        +-------------------------------+
        [3 rows x 1 columns]


        # Save the transformer.
        >>> hasher.save('save-path')
    '''

    _fit_examples_doc = _fit_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc
    _transform_examples_doc = _transform_examples_doc


    get_default_options = staticmethod(_get_default_options_wrapper(
        '_FeatureHasher', 'toolkits.feature_engineering._feature_hasher',
                                                    'FeatureHasher', True))

    def __init__(self, features=None, excluded_features=None, num_bits=18,
                                        output_column_name='hashed_features'):

        # Process and make a copy of the features, exclude.
        _features, _exclude = _internal_utils.process_features(
                                        features, excluded_features)

        # Type checking
        _raise_error_if_not_of_type(num_bits, [int])
        _raise_error_if_not_of_type(output_column_name, [str])

        # Set up options
        opts = {
            'num_bits': num_bits,
            'output_column_name': output_column_name,
            }
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features

        # Initialize object
        proxy = _gl.extensions._FeatureHasher()
        proxy.init_transformer(opts)
        super(FeatureHasher, self).__init__(proxy, self.__class__)

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
            ("Number of bits", 'num_bits')
        ]
        section_titles = [ 'Model fields' ]

        return ([fields], section_titles)

    def __repr__(self):

        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width= 30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        hasher = _gl.feature_engineering.FeatureHasher(features = ['a', 'b'])
        return hasher.fit(sf), sf
