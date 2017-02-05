import graphlab as _gl
import copy as _copy
from graphlab.toolkits.feature_engineering._feature_engineering import Transformer
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _precomputed_field
from graphlab.util import _raise_error_if_not_of_type

from . import  _internal_utils
from ._doc_utils import republish_docs

NoneType = type(None)


_fit_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,3], 'b' : [2,3,4]})

            # Create a quadratic feature space using all pairs of features in ['a', 'b'].
            >>> quad =  graphlab.feature_engineering.QuadraticFeatures(
                                features = ['a', 'b'])
            >>> quad =  quad.fit(sf)
'''


_fit_transform_examples_doc = '''
            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,3], 'b' : [2,3,4]})


            # Create a quadratic feature space using all pairs of features in ['a', 'b'].
            >>> quad =  graphlab.feature_engineering.QuadraticFeatures(
                                    features = ['a', 'b'])

            # Transform the space into a new space (called "quadratic_features")
            >>> quadratic_sf = quad.fit_transform(sf)

            Columns:
              a  int
              b  int
              quadratic_features  dict

            Rows: 3

            Data:
            +---+---+-------------------------------+
            | a | b |       quadratic_features      |
            +---+---+-------------------------------+
            | 1 | 2 | {'a, b': 2, 'a, a': 1, 'b,... |
            | 2 | 3 | {'a, b': 6, 'a, a': 4, 'b,... |
            | 3 | 4 | {'a, b': 12, 'a, a': 9, 'b... |
            +---+---+-------------------------------+
            [3 rows x 3 columns]

'''

_transform_examples_doc = '''

            # Numeric data.
            # -----------------------------------------------------------------

            # Create the data
            >>> sf = graphlab.SFrame({'a' : [1,2,3], 'b' : [2,3,4]})

            # Hash the space of columns ['a', 'b'] into a single space.
            >>> quad =  graphlab.feature_engineering.QuadraticFeatures(
                            features = ['a', 'b'])
            >>> quad =  quad.fit(sf)

            # Transform the space into the hashed space (called "quadratic_features")
            >>> quadratic_sf = quad.transform(sf)

            Columns:
              a  int
              b  int
              quadratic_features  dict

            Rows: 3

            Data:
            +---+---+-------------------------------+
            | a | b |       quadratic_features      |
            +---+---+-------------------------------+
            | 1 | 2 | {'a, b': 2, 'a, a': 1, 'b,... |
            | 2 | 3 | {'a, b': 6, 'a, a': 4, 'b,... |
            | 3 | 4 | {'a, b': 12, 'a, a': 9, 'b... |
            +---+---+-------------------------------+
            [3 rows x 3 columns]


            # String/Categorical data
            # -----------------------------------------------------------------

            # Create the data
            >>> sf = graphlab.SFrame({'a' : ['a','b','c'], 'b' : ['d','e','f']})

            # Hash the feature space ['a', 'b']
            >>> quad =  graphlab.feature_engineering.QuadraticFeatures(
                            features = ['a', 'b'])
            >>> quad =  quad.fit(sf)

            # Transform the data into the hashed space (called "quadratic_features").
            >>> quadratic_sf = quad.transform(sf)

            Columns:
              a  str
              b  str
              quadratic_features  dict

            Rows: 3

            Data:
            +---+---+-------------------------------+
            | a | b |       quadratic_features      |
            +---+---+-------------------------------+
            | a | d | {'a:a, a:a': 1, 'a:a, b:d'... |
            | b | e | {'b:e, b:e': 1, 'a:b, a:b'... |
            | c | f | {'b:f, b:f': 1, 'a:c, a:c'... |
            +---+---+-------------------------------+
            [3 rows x 3 columns]


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
            >>> quad =  graphlab.feature_engineering.QuadraticFeatures(
                                        features = 'categories')
            >>> quad =  quad.fit(sf)

            # Transform the data into the hashed space.
            >>> quadratic_sf = quad.transform(sf)

            Columns:
              categories  list
              quadratic_features  dict

            Rows: 6

            Data:
            +-----------------+-------------------------------+
            |    categories   |       quadratic_features      |
            +-----------------+-------------------------------+
            |  [cat, mammal]  | {'categories:1:mammal, cat... |
            |  [cat, mammal]  | {'categories:1:mammal, cat... |
            | [human, mammal] | {'categories:1:mammal, cat... |
            | [seahawk, bird] | {'categories:1:bird, categ... |
            |   [duck, bird]  | {'categories:1:bird, categ... |
            | [seahawk, bird] | {'categories:1:bird, categ... |
            +-----------------+-------------------------------+
            [6 rows x 2 columns]


            # Dictionary data
            # -----------------------------------------------------------------
            >>> sf = graphlab.SFrame({'attributes':
                            [{'height':'tall', 'age': 'senior', 'weight': 'thin'},
                             {'height':'short', 'age': 'child', 'weight': 'thin'},
                             {'height':'giant', 'age': 'adult', 'weight': 'fat'},
                             {'height':'short', 'age': 'child', 'weight': 'thin'},
                             {'height':'tall', 'age': 'child', 'weight': 'fat'}]})
            # Hash the feature set
            >>> quad =  graphlab.feature_engineering.QuadraticFeatures(
                                        features = 'attributes')
            >>> quad =  quad.fit(sf)

            # Transform the data into the hashed space.
            >>> quadratic_sf = quad.transform(sf)

           Columns:
             attributes  dict
             quadratic_features  dict

           Rows: 5

           Data:
           +-------------------------------+-------------------------------+
           |           attributes          |       quadratic_features      |
           +-------------------------------+-------------------------------+
           | {'age': 'senior', 'weight'... | {'attributes:age:senior, a... |
           | {'age': 'child', 'weight':... | {'attributes:height:short,... |
           | {'age': 'adult', 'weight':... | {'attributes:height:giant,... |
           | {'age': 'child', 'weight':... | {'attributes:height:short,... |
           | {'age': 'child', 'weight':... | {'attributes:weight:fat, a... |
           +-------------------------------+-------------------------------+
           [5 rows x 2 columns]
'''

@republish_docs
class QuadraticFeatures(Transformer):
    '''
    Calculates quadratic interaction terms between features.

    Adding interaction terms is a good way of injecting complex relationships
    between predictor variables while still using a simple learning algorithm
    (ie. Logistic Regression) that is easy to use and explain. The QuadraticFeatures
    transformer accomplishes this by taking a row of the SFrame, and multiplying
    the specified features together. If the features are of array.array or dictionary
    type, multiplications of all possible pairs are computed. If a non-numeric
    value is encountered, 1 is substituted for the value and the old string
    value becomes part of the interaction term name. Supported types are int,
    float, string, array.array, list, and dict.

    When the transformer is applied, an additional column with name
    specified by 'output_column_name' is added to the input SFrame.
    In this column of dictionary type, interactions are specified in the
    key names (by concatenating column names and keys/indices if applicable)
    and values are the multiplied values.

    Parameters
    ----------
    features : list | str | tuple , optional
        Can be a list of tuples, a list of feature name strings, a
        feature name string, a tuple, or None. If it is a
        list of tuples containing two interaction terms, those are the
        calculated interaction terms. In the case of providing a list of
        feature_names, all pairs between those feature names are calculated.
        If the list is of size none, all feature pairs are calculated in the
        SFrame the transformer is applied to.

    excluded_features: list | str | tuple, optional
        Can be a list of tuples, a list of feature name strings, a
        feature name string, a tuple, or None. In the case
        of tuples, those particular interactions are excluded. In the case
        of feature names, all interactions with those features are excluded.
        Cannot set both 'exclude' and 'features'.

    output_column_name : str , optional
        The name of the output column

    Returns
    -------
    out : QuadraticFeatures
        A QuadraticFeatures object which is initialized with the defined
        parameters.

    See Also
    --------
    graphlab.toolkits.feature_engineering.QuadraticFeatures
    graphlab.toolkits.feature_engineering.create

    Examples
    --------

    .. sourcecode:: python

        from graphlab.toolkits.feature_engineering import *

        # Construct a quadratic features transformer with default options.
        >>> sf = graphlab.SFrame({'a': [1,2,3], 'b' : [2,3,4], 'c': [9,10,11]})
        >>> quadratic = graphlab.feature_engineering.create(sf,
                    QuadraticFeatures(features = ['a', 'b', 'c']))

        # Transform the data.
        >>> quadratic_sf = quadratic.transform(sf)

        # Save the transformer.
        >>> quadratic.save('save-path')

    '''

    _fit_examples_doc = _fit_examples_doc
    _transform_examples_doc = _transform_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc

    get_default_options = staticmethod(_get_default_options_wrapper(
        '_QuadraticFeatures', 'toolkits.feature_engineering._quadratic_features', 'QuadraticFeatures', True))

    def __init__(self, features=None, excluded_features=None, output_column_name='quadratic_features'):

        #Type checking
        _raise_error_if_not_of_type(output_column_name, [str])

        # set up options
        opts = {
            'output_column_name': output_column_name
        }
        # Make a copy of the parameters.
        _features = _copy.copy(features)
        _exclude = _copy.copy(excluded_features)


        # Check of both are None or empty.
        if _features and _exclude:
            raise ValueError("The parameters 'features' and 'exclude' cannot both be set."
                    " Please set one or the other.")
        if _features == [] and not _exclude:
            raise ValueError("Features cannot be an empty list.")

        # Check types
        _raise_error_if_not_of_type(_features, [NoneType, list, str, tuple], 'features')
        _raise_error_if_not_of_type(_exclude, [NoneType,  list, str, tuple], 'exclude')

        # Allow a single list
        _features = [_features] if type(_features) == str or type(_features) == tuple else _features
        _exclude = [_exclude] if type(_exclude) == str or type(_exclude) == tuple else _exclude


        # Type check each feature/exclude
        if _features:
            for f in _features:
                _raise_error_if_not_of_type(f, [str, tuple], "Feature names")
        if _exclude:
            for e in _exclude:
                _raise_error_if_not_of_type(e, [str, tuple], "Excluded feature names")

        if _exclude:
            opts['exclude'] = True
            unprocessed_features = _exclude
        else:
            opts['exclude'] = False
            unprocessed_features = _features

        pair_list = set()

        if unprocessed_features is not None:
            if type(unprocessed_features[0]) is tuple:
                for t in unprocessed_features:
                    pair_list.add(tuple(sorted(t)))
            elif type(unprocessed_features[0]) is str:
                if _exclude:
                    for t in unprocessed_features:
                        pair_list.add(t)
                else:
                    for t in unprocessed_features:
                        for k in unprocessed_features:
                            pair_list.add(tuple(sorted((t, k))))

        if type(output_column_name) is not str:
            raise ValueError("'output_column_name' must be of type str")

        if unprocessed_features is not None:
            if type(unprocessed_features[0]) is str:
                opts['features'] = unprocessed_features
                if _exclude:
                    opts['feature_pairs'] = list(pair_list)
                else:
                    opts['feature_pairs'] = [list(x) for x in pair_list]
            else:
                opts['feature_pairs'] = [list(x) for x in pair_list ]
                opts['features'] = [list(x) for x in unprocessed_features]
        else:
            opts['feature_pairs'] = None
            opts['features'] = None


        # initialize object
        proxy = _gl.extensions._QuadraticFeatures()
        proxy.init_transformer(opts)
        super(QuadraticFeatures, self).__init__(proxy, self.__class__)

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
            ("Output column name", 'output_column_name')
        ]
        section_titles = [ 'Model fields' ]

        return ([fields], section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        encoder = _gl.feature_engineering.QuadraticFeatures(
            features = ['a', 'b'])
        return encoder.fit(sf), sf
