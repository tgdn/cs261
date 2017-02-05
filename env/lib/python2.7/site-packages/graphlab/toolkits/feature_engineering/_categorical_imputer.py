import graphlab as _gl
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits.feature_engineering._feature_engineering import Transformer
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _precomputed_field
from graphlab.util import _raise_error_if_not_of_type

from . import _internal_utils
from ._doc_utils import republish_docs

_fit_examples_doc = '''
            # Impute the column "feature" using information from columns ['a', 'b']
            >>> sf = graphlab.SFrame({'a' : [1,1,1], 'b' : [1,0,1], 'feature' : [1,2,None]})
            >>> imputer = graphlab.feature_engineering.CategoricalImputer(
                          feature = 'feature', reference_features = ['a', 'b'])
            >>> fit_imputer = imputer.fit(sf)
'''

_fit_transform_examples_doc = '''
            # Impute the column "feature" using information from columns ['a', 'b']
            >>> sf = graphlab.SFrame({'a' : [1,1,1], 'b' : [1,0,1], 'label' : [1,2,None]})
            >>> imputer = graphlab.feature_engineering.CategoricalImputer(
                                 feature = 'label', reference_features = ['a', 'b'])
            >>> imputed_sf = imputer.fit_transform(sf)
            PROGRESS: Initializing data.
            ...
            >>> imputed_sf
            Columns:
                a    int
                b    int
                label    int
                predicted_feature_label    int
                feature_probability_label    float

            Rows: 3

            Data:
            +---+---+---------+-------------------------+---------------------------+
            | a | b | feature | predicted_feature_label | feature_probability_label |
            +---+---+---------+-------------------------+---------------------------+
            | 1 | 1 |    1    |            1            |            1.0            |
            | 1 | 0 |    2    |            2            |            1.0            |
            | 1 | 1 |   None  |            1            |            1.0            |
            +---+---+---------+-------------------------+---------------------------+
            [3 rows x 5 columns]
'''

_transform_examples_doc = '''
            # Impute the column "feature" using information from columns ['a', 'b']
            >>> sf = graphlab.SFrame({'a' : [1,1,1], 'b' : [1,0,1], 'label' : [1,2,None]})
            >>> imputer = graphlab.feature_engineering.CategoricalImputer(
                                 feature = 'label', reference_features = ['a', 'b'])
            >>> fit_imputer = imputer.fit(sf)
            >>> imputed_sf = fit_imputer.transform(sf)
            PROGRESS: Initializing data.
            ...
            ...
            >>> imputed_sf
            Columns:
                a    int
                b    int
                label    int
                predicted_feature_label    int
                feature_probability_label    float

            Rows: 3

            Data:
            +---+---+---------+-------------------------+---------------------------+
            | a | b | feature | predicted_feature_label | feature_probability_label |
            +---+---+---------+-------------------------+---------------------------+
            | 1 | 1 |    1    |            1            |            1.0            |
            | 1 | 0 |    2    |            2            |            1.0            |
            | 1 | 1 |   None  |            1            |            1.0            |
            +---+---+---------+-------------------------+---------------------------+
            [3 rows x 5 columns]
'''

@republish_docs
class CategoricalImputer(Transformer):
    '''
    The purpose of this imputer is to fill missing values (None) in data sets
    that have categorical data. For instance, if the data set has a "feature" column
    where some rows have values, and some rows have None, this imputer will fill
    the Nones with values. It will also return a probability associated with
    the imputed value.

    This is accomplished by grouping the data based on provided reference_features
    (unsupervised clustering) then by assigning reference_features to the clusters following
    a graph walk among the resulting clusters.

    Parameters
    ----------
    reference_features : list[str] , optional
        Column names of reference_features to be used for clustering. If None, all columns are
        selected.

    feature : 'feature', optional
        Name of the column to impute. This column should contain some categorical
        values, as well as rows with None. Those rows will be imputed.


    Returns
    -------
    out : CategoricalImputer
        A CategoricalImputer object which is initialized with the defined
        parameters.

    See Also
    --------
    graphlab.toolkits.feature_engineering._categorical_imputer.CategoricalImputer
    graphlab.toolkits.feature_engineering.create

    Examples
    --------

    .. sourcecode:: python

        from graphlab.toolkits.feature_engineering import *

        # Impute the column "feature" using information from columns ['a', 'b']
        >>> sf = graphlab.SFrame({'a' : [0,1,1], 'b' : [1,0,0], 'label' : [1,2,None]})
        >>> imputer = graphlab.feature_engineering.CategoricalImputer(
                             feature = 'label', reference_features = ['a', 'b'])
        >>> imputer.fit(sf)

        # Print the input data.
        >>> sf
        Columns:
        a    int
        b    int
        label    int

        Rows: 3

        Data:
        +---+---+-------+
        | a | b | label |
        +---+---+-------+
        | 0 | 1 |   1   |
        | 1 | 0 |   2   |
        | 1 | 0 |  None |
        +---+---+-------+
        [3 rows x 3 columns]

        # Transform the data using the imputer.
        >>> imputed_sf = imputer.transform(sf)

        # Retrieve the imputed data.
        >>> imputed_sf
        Columns:
            a    int
            b    int
            label    int
            predicted_feature_label    int
            feature_probability_label    float

        Rows: 3

        Data:
        +---+---+---------+-------------------------+---------------------------+
        | a | b | feature | predicted_feature_label | feature_probability_label |
        +---+---+---------+-------------------------+---------------------------+
        | 0 | 1 |    1    |            1            |            1.0            |
        | 1 | 0 |    2    |            2            |            1.0            |
        | 1 | 0 |   None  |            2            |            1.0            |
        +---+---+---------+-------------------------+---------------------------+
        [3 rows x 5 columns]

        # Save the transformer.
        >>> imputer.save('save-path')

        # Bin only a single column 'a'.
        >>> imputer = graphlab.feature_engineering.create(sf,
                graphlab.feature_engineering.CategoricalImputer(
                    reference_features = ['a'], feature='label'))


    '''

    _fit_examples_doc = _fit_examples_doc
    _transform_examples_doc = _transform_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc



    get_default_options = staticmethod(_get_default_options_wrapper(
            '_CategoricalImputer', 'toolkits.feature_engineering._categorical_imputer', 'CategoricalImputer', True))

    def __init__(self, reference_features=None, feature="feature", verbose=False):

        # Process and make a copy of the reference_features
        _reference_features, _exclude = _internal_utils.process_features(reference_features, None)

        # Type checking
        _raise_error_if_not_of_type(feature, [str])

        # Set up options
        opts = {
          'reference_features': reference_features,
          'feature': feature,
          'verbose': verbose
        }
        opts['reference_features'] = _reference_features

        # Initialize object
        proxy = _gl.extensions._CategoricalImputer()
        proxy.init_transformer(opts)
        super(CategoricalImputer, self).__init__(proxy, self.__class__)

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
                  ('<feature>','<field>')
        section_titles: list
            A list of section titles.
              The order matches that of the 'sections' object.
        """
        _reference_features = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('reference_features')))

        fields = [
            ("reference_features", _reference_features),
            ("Column to impute", 'feature')
        ]
        section_titles = ['Model fields']

        return ([fields], section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a' : [1, 1, 1],
                         'b' : [1, 0, 1],
                         'feature' : [1, 2, None]})
        imputer = _gl.feature_engineering.CategoricalImputer(
            feature = 'feature', reference_features = ['a', 'b'])
        return imputer.fit(sf), sf
