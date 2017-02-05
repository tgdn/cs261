import graphlab.connect as _mt
import graphlab as _gl

from ._spacy_transformer import SpacyTransformer as _SpacyTransformer

from graphlab.toolkits._model import PythonProxy as _PythonProxy
from graphlab.toolkits._internal_utils import _precomputed_field

from graphlab.toolkits._internal_utils import _check_categorical_option_type

from . import _internal_utils


class SentenceSplitter(_SpacyTransformer):
    """
    The SentenceSplitter takes SFrame columns of type string or list,
    and transforms into list of strings, where each element is a single sentence.
    If the input column type is list, each element must either be a string or NoneType.

    .. note:: Depends on spaCy
        This transformer depends on spaCy, a Python package for natural language
        processing. Please see spacy.io for installation information.

    Parameters
    ----------
    features : list[str] , optional
        Column names of features to be transformed. If None, all columns are
        selected. Features must be of type str, list[str].

    excluded_features : list[str] | str | None, optional
        Column names of features to be ignored in transformation. Can be string
        or list of strings. Either 'excluded_features' or 'features' can be
        passed, but not both.

    verbose: bool, optional
        When True, prints progress of transformation.

    output_column_prefix : str, optional
        The prefix to use for the column name of each transformed column.
        When provided, the transformation will add columns to the input data,
        where the new name is "`output_column_prefix`.original_column_name".
        If `output_column_prefix=None` (default), then the output column name
        is the same as the original feature column name.

    See Also
    --------
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

        >>> import graphlab as gl

        # Create data.
        >>> sf = gl.SFrame({
        ...    'text': ['This is sentence 1. This is sentence two.']})

        # Create a SentenceSplitter transformer.
        >>> from graphlab.feature_engineering import SentenceSplitter
        >>> transformer = SentenceSplitter()

        # Fit and transform the data.
        >>> transformed_sf = transformer.fit_transform(sf)
        Columns:
            text    list
        +-------------------------------+
        |              text             |
        +-------------------------------+
        | [This is sentence 1., This... |
        +-------------------------------+
        [1 rows x 1 columns]


        # For SFrame of type list
        >>> import graphlab as gl

        # Create data.
        >>> sf = gl.SFrame({
        ...    'text': [['This is sentence 1. This is sentence two.']]})

        # Create a SentenceSplitter transformer.
        >>> from graphlab.feature_engineering import SentenceSplitter
        >>> transformer = SentenceSplitter()

        # Fit and transform the data.
        >>> transformed_sf = transformer.fit_transform(sf)
        +-------------------------------+
        |              text             |
        +-------------------------------+
        | [This is sentence 1., This... |
        +-------------------------------+
        [1 rows x 1 columns]


    """

    @staticmethod
    def get_default_options(output_type = 'sframe'):
        """
        Return information about the default options.

        Parameters
        ----------
        output_type : str, optional

            The output can be of the following types.

            - `sframe`: A table description each option used in the model.
            - `json`: A list of option dictionaries.


        Returns
        -------
        out : SFrame
            Each row in the output SFrames correspond to a parameter, and includes
            columns for default values, lower and upper bounds, description, and
            type.
        """
        _check_categorical_option_type('output_type', output_type,
                                    ['json', 'sframe'])
        out = _gl.SFrame({
            'name': ['features','excluded_features', 'verbose','output_column_prefix'],
            'default_value': ['None', 'None', 'True', 'None'],
                'parameter_type': ['list[str]', 'list[func]','bool', 'str'],
            'lower_bound': ['None', 'None', 'None','None'],
            'upper_bound': ['None', 'None', 'None','None'],
            'description': ['Features to include in transformation.',
                            'Features to exclude from transformation.',
                            'Verbosity flag',
                            'Prefix of the output column.']})
        if output_type == "sframe":
            return out
        else:
            return {row['name']:{"default_value":row['default_value'],
                "description": row['description'],
                "upper_bound": row['upper_bound'],
                "lower_bound": row['lower_bound'],
                "parameter_type": row['parameter_type']} for row in out}



    _SENTENCE_SPLITTER_VERSION = 0

    def _get_version(self):
        return self._SENTENCE_SPLITTER_VERSION

    def __init__(self, features=None, excluded_features=None, 
                 output_column_prefix=None, verbose=True):
        
        super(SentenceSplitter, self).__init__(features, excluded_features,
                                               output_column_prefix, verbose)

    def fit(self, dataset):
        """
        Fits a transformer using the SFrame `data`.

        Parameters
        ----------
        data : SFrame
            The data used to fit the transformer.

        Returns
        -------
        self (A fitted object)

        See Also
        --------
        transform, fit_transform

        Examples
        --------

        .. sourcecode:: python

            >>> import graphlab as gl

            # Create data.
            >>> sf = gl.SFrame({
            ...    'text': ['This is sentence 1. This is sentence two.']})

            # Create a SentenceSplitter transformer.
            >>> from graphlab.feature_engineering import SentenceSplitter
            >>> transformer = SentenceSplitter()

            # Fit and transform the data.
            >>> transformer.fit(sf)
            >>> transformer['feature_columns']
            ['text']

        """
        return super(SentenceSplitter, self).fit(dataset)

    def transform(self, dataset):
        """
        Transform the SFrame `data` using a fitted model.

        Parameters
        ----------
        data : SFrame
            The data  to be transformed.

        Returns
        -------
        A transformed SFrame.

        Returns
        -------
        out: SFrame
            A transformed SFrame.

        See Also
        --------
        fit, fit_transform

        Examples
        --------

        .. sourcecode:: python

            >>> import graphlab as gl

            # Create data.
            >>> sf = gl.SFrame({
            ...    'text': ['This is sentence 1. This is sentence two.']})

            # Create a SentenceSplitter transformer.
            >>> from graphlab.toolkits.feature_engineering import SentenceSplitter
            >>> transformer = SentenceSplitter()

            # Fit and transform the data.
            >>> transformer.fit(sf)
            >>> transformed_sf = transformer.transform(sf)
            +-------------------------------+
            |              text             |
            +-------------------------------+
            | [This is sentence 1., This... |
            +-------------------------------+
            [1 rows x 1 columns]
        """
        return super(SentenceSplitter,self).transform(dataset)

    def _get_summary_struct(self):
        _features = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('features')))
        _exclude = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('excluded_features')))

        fields = [
            ("Features", _features),
            ("Excluded_features", _exclude),
            ("Output column prefix", 'output_column_prefix'),
        ]
        section_titles = ['Model fields']

        return ([fields], section_titles)

    def _get_column_type(self):
        return list

    '''
    False means spaCy english object doesn't apply tagger, entity, or parser
    '''
    def _get_to_tag(self):
        return True

    '''
    False means spaCy english object applies tagger and entity, not parser
    '''
    def _get_to_parse(self):
        return True

    '''
    False means spaCy english object applies tagger and parser, not enity.
    '''
    def _get_apply_named_entity(self):
        return False

    def _transform_column_impl(self, sb, i, doc, none_index):
        """
        Helper function that performs a transformation on the given doc
        and updates the provided SFrameBuilder object.

        Parameters
        ----------
        sb : SFrameBuilder

        i : int
            The row number of the current doc.

        doc : spaCy.Document
            A document object.

        none_index : deque
            A deque of row numbers that contain Nones.

        Returns
        -------
            Modifies sb and none_index in place.
        """
        if len(none_index) > 0 and i == none_index[0]:
            sb.append([None])
            none_index.popleft()
        elif len(doc) == 0:
            sb.append([[]])
        else:
            sents = [s.text for s in doc.sents]
            sb.append([sents])

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame(
            {'docs': ['This is the first sentence. This is the second sentence.']})
        encoder = _gl.feature_engineering.SentenceSplitter('docs')
        encoder = encoder.fit(sf)
        return encoder, sf
