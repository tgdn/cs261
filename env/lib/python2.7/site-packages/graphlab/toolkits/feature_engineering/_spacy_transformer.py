import graphlab.connect as _mt
import copy as _copy
import graphlab as _gl
from collections import deque
import sys as _sys

from graphlab.toolkits._model import PythonProxy as _PythonProxy
from graphlab.data_structures.sframe_builder import SFrameBuilder
from graphlab.toolkits.feature_engineering._internal_utils import *
from graphlab.toolkits.feature_engineering import TransformerBase as _TransformerBase
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.data_structures.sframe import SFrame as _SFrame
from graphlab.util import _raise_error_if_not_of_type
from graphlab.util import sys_info as _sys_info

from ._internal_utils import _import_spacy

NoneType = type(None)

class SpacyTransformer(_TransformerBase, _ProxyBasedModel):
    """
    A base class for transformers based on Python and using Spacy.
    """
    _nlp = None

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        A function to load a previously saved SentenceSplitter instance.

        Parameters
        ----------
        unpickler : GLUnpickler
            A GLUnpickler file handler.

        version : int
            Version number maintained by the class writer.
        """

        _mt._get_metric_tracker().track(cls.__name__ + '.load_version')

        state, _exclude, _features = unpickler.load()

        model = cls.__new__(cls)
        model._setup()
        model.__proxy__.update(state)
        model._exclude = _exclude
        model._features = _features

        return model


    def _setup(self):
        """
        Sets up the model; common between __init__ and load.
        """
        self.__proxy__ = _PythonProxy()

        #Try importing spacy
        if SpacyTransformer._nlp is None:
            SpacyTransformer._nlp = _import_spacy()


    def __init__(self, features = None, excluded_features = None,
                 output_column_prefix = None, verbose = True):

        self._setup()

        _features, _exclude = process_features(features, excluded_features)

        #Type check
        _raise_error_if_not_of_type(output_column_prefix, [str, NoneType])
        _raise_error_if_not_of_type(verbose, [bool])

        state = {}
        state['output_column_prefix'] = output_column_prefix
        state['features'] = _features
        state['excluded_features'] = _exclude
        state['fitted'] = False
        state['verbose'] = verbose

        if _exclude:
            self._exclude = True
            self._features = _exclude
        else:
            self._exclude = False
            self._features = _features

        self.__proxy__.update(state)


    @staticmethod
    def get_default_options(output_type = 'sframe'):
        raise NotImplementedError()

    def _get_version(self):
        raise NotImplementedError()

    def fit(self, dataset):
        """
        Fits a transformer using the SFrame `dataset`.

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
        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.fit')

        _raise_error_if_not_of_type(dataset, [_SFrame])

        fitted_state = {}
        feature_columns = get_column_names(dataset, self._exclude, self._features)
        feature_columns = select_valid_features(dataset, feature_columns, [str, list])
        fitted_state['features'] = feature_columns
        validate_feature_columns(dataset.column_names(), feature_columns)

        fitted_state['col_type_map'] = {col_name: col_type for (col_name, col_type) in zip(dataset.column_names(), dataset.column_types())}

        fitted_state['fitted'] = True

        self.__proxy__.update(fitted_state)

        return self

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
        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.transform')

        if ( not self.__proxy__['fitted']):
            raise RuntimeError("The SentenceSplitter must be fitted before .transform() is called.")

        dataset_copy = _copy.copy(dataset)

        transform_features = select_feature_subset(dataset, self.__proxy__['features'])
        validate_feature_types(transform_features, self.__proxy__['col_type_map'], dataset)

        if self.__proxy__['output_column_prefix'] == None:
            output_column_prefix = ""
        else:
            output_column_prefix = self.__proxy__['output_column_prefix'] + "."

        for f in transform_features:
            output_column_name = output_column_prefix + f
            dataset_copy[output_column_name] = self._transform_column(dataset_copy[f], f)

        return dataset_copy

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and train the model.

        Returns
        -------
        out : dict
            Dictionary with options used to define and train the model.

        Examples
        --------
        >>> options = m.get_current_options()
        """

        return {k: self.__proxy__[k] for k in self.__class__.get_default_options()['name']}

    def _get_summary_struct(self):
        raise NotImplementedError()

    def summary(self, output=None):
        """
        Print a summary of the model.
        The summary includes a description of training
        data, options, hyper-parameters, and statistics measured during model
        creation.

        Examples
        --------
        >>> m.summary()

        Parameters
        ----------
        output : string, None
            The type of summary to return.
            None or 'stdout' : prints directly to stdout
            'str' : string of summary
            'dict' : a dict with 'sections' and 'section_titles' ordered lists.
            The entries in the 'sections' list are tuples of the form
            ('label', 'value').
        """
        if output is None or output == 'stdout':
            pass
        elif (output == 'str'):
            return self.__repr__()
        elif output == 'dict':
            return _toolkit_serialize_summary_struct( self, \
                                            *self._get_summary_struct() )
        try:
            return self.__repr__()
        except:
            return self.__class__.__name__

    def _transform_column(self, sa, column_name):
        if self.__proxy__['verbose']:
            print("PROGRESS: Transforming column " + column_name)
        sys_info = _sys_info.get_sys_info()
        num_cpus = sys_info['num_cpus']
        sb = SFrameBuilder([self._get_column_type()])

        # Decode text, and keep track of rows that contain None values.
        none_index = deque()

        def spacy_generator(sa):

            def as_unicode(x):
                if _sys.version_info.major == 3:
                    return x
                else:
                    try:
                        element = unicode(x.decode('utf-8'))
                        return element
                    except:
                        raise ValueError("Input data in column " + column_name +
                                         " cannot be decoded as 'utf-8'. Please encode" +
                                         " as 'utf-8'.")
            def raise_type_error():
                raise ValueError("All elements in input SArray must be of" +
                             " type str, None, or list. Lists must only"+
                             " contain elements of String or NoneType")

            #Iterate through SArray
            for t, x in enumerate(sa.__iter__()):
                # Placeholder element has to be generated
                if x is None:
                    none_index.append(t)
                    yield as_unicode('')
                #Iterate through lists, ignore None's, and fail on other types
                elif type(x) == list:
                    to_yield = ''
                    for r in x:
                        if type(r) == str:
                            to_yield = to_yield + ' ' + r
                        elif r is None:
                            pass
                        else:
                            raise_type_error()
                    yield as_unicode(to_yield)
                elif type(x) == str:
                    yield as_unicode(x)
                else:
                    raise_type_error()

        text = spacy_generator(sa)

        to_tag = self._get_to_tag()
        to_parse = self._get_to_parse()
        apply_named_entity = self._get_apply_named_entity()

        for i, doc in enumerate(self._nlp.pipe(text, batch_size=100, n_threads=num_cpus, tag=to_tag, parse=to_parse, entity=apply_named_entity)):
            self._transform_column_impl(sb, i, doc, none_index)
            if i % 1000 == 0 and i > 0 and self.__proxy__['verbose']:
                print('PROGRESS: {} of {}'.format(i, sa.size()))

        sf = sb.close()
        return sf['X1']

    '''
    Returns bool. False means spaCy english object doesn't apply tagger, entity,
    or parser
    '''
    def _get_to_tag(self):
        raise NotImplementedError()

    '''
    Returns bool. False means spaCy english object applies tagger and entity,
    not parser
    '''
    def _get_to_parse(self):
        raise NotImplementedError()

    '''
    Returns Bool. False means spaCy english object applies tagger and parser,
    not enity.
    '''
    def _get_apply_named_entity(self):
        raise NotImplementedError()

    def _get_column_type(self):
        raise NotImplementedError()

    def _transform_column_impl(self, sb, i, doc, none_index):
        raise NotImplementedError()

    def _save_impl(self, pickler):
        """
        Save the model as a directory, which can be loaded with the
        :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        pickler : GLPickler
            An opened GLPickle archive (Do not close the archive).

        See Also
        --------
        graphlab.load_model

        Examples
        --------
        >>> model.save('my_model_file')
        >>> loaded_model = graphlab.load_model('my_model_file')
        """
        _mt._get_metric_tracker().track(self.__module__ + '.save_impl')
        state = self.__proxy__.state
        pickler.dump( (state, self._exclude, self._features))

    @classmethod
    def _get_instance_and_data(cls):
        raise NotImplementedError()
