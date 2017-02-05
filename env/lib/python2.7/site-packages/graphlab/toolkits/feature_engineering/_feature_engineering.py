import graphlab as _gl
from graphlab.util import _make_internal_url
import graphlab.connect.main as glconnect
import graphlab.connect as _mt

from graphlab.toolkits._model import CustomModel
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits._internal_utils import _toolkit_repr_print, \
                                                _precomputed_field, \
                                                _raise_error_if_not_sframe

# Base class for models written by users
# ---------------------------------------------------
class TransformerBase(CustomModel):
    """
    An abstract base class for user defined Transformers.

    **Overview**

    A Transformer is a stateful object that transforms input data (as an
    SFrame) from one form to another. Transformers are commonly used for
    feature engineering. In addition to the modules provided in GraphLab
    Create, users can extend the following class and write transformers that
    integrate seamlessly with the already existing ones.

    **Defining Custom Transformers**

    Each transformer object is one must have the following methods:

        +---------------+---------------------------------------------------+
        |   Method      | Description                                       |
        +===============+===================================================+
        | __init__      | Construct the object.                             |
        +---------------+---------------------------------------------------+
        | fit           | Fit the object using training data.               |
        +---------------+---------------------------------------------------+
        | transform     | Transform the object on training/test data.       |
        +---------------+---------------------------------------------------+

    In addition to these methods, there are convenience methods with default
    implementations:

        +---------------+---------------------------------------------------+
        |   Method      | Description                                       |
        +===============+===================================================+
        | save          | Save the model to a GraphLab Create archive.      |
        +---------------+---------------------------------------------------+
        | fit_transform | First perform fit() and then transform() on data. |
        +---------------+---------------------------------------------------+

    See Also
    --------
    :class:`graphlab.toolkits.feature_engineering.TransformerChain`,
    :func:`graphlab.toolkits.feature_engineering.create`

    Notes
    ------
    - User defined Transformers behave identically to those that are already
      provided. They can be saved/loaded both locally and remotely, can
      be chained together, and can be deployed as components of predictive
      services.

    Examples
    --------

    In this example, we will write a simple Transformer that will subtract
    (for each column) the mean value observed during the `fit` stage.

    .. sourcecode:: python

        import graphlab
        from graphlab.toolkits.feature_engineering import TransformerBase

        class MyTransformer(TransformerBase):

            def __init__(self):
                pass

            def fit(self, dataset):
                ''' Learn means during the fit stage.'''
                self.mean = {}
                for col in dataset.column_names():
                    self.mean[col] = dataset[col].mean()
                return self

            def transform(self, dataset):
                ''' Subtract means during the transform stage.'''
                new_dataset = graphlab.SFrame()
                for col in dataset.column_names():
                    new_dataset[col] = dataset[col] - self.mean[col]
                return new_dataset

        # Create the model
        model = gl.feature_engineering.create(dataset, MyTransformer())

        # Transform new data
        transformed_sf = model.transform(sf)

        # Save and load this model.
        model.save('foo-bar')
        loaded_model = graphlab.load_model('foo-bar')

    """

    def __init__(self, **kwargs):
        pass

    def _get_summary_struct(self):
        model_fields = []
        for attr in self.__dict__:
            if not attr.startswith('_'):
                model_fields.append((attr, _precomputed_field(getattr(self, attr))))

        return ([model_fields], ['Attributes'] )

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=20)

    def fit(self, data):
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

            my_tr = MyTransformer(features = ['salary', 'age'])
            my_tr = mt_tr.fit(sf)
        """
        pass

    def transform(self, data):
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

            my_tr = graphlab.feature_engineering.create(train_data,
                                                        MyTransformer())
            transformed_sf = my_tr.transform(sf)
        """
        raise NotImplementedError

    def fit_transform(self, data):
        """
        First fit a transformer using the SFrame `data` and then return a transformed
        version of `data`.

        Parameters
        ----------
        data : SFrame
            The data used to fit the transformer. The same data is then also
            transformed.

        Returns
        -------
        Transformed SFrame.

        See Also
        --------
        transform, fit_transform

        Notes
        -----
        The default implementation calls `fit` and then calls `transform`.
        You may override this function with a more efficient implementation."

        Examples
        --------
        .. sourcecode:: python

            my_tr = MyTransformer()
            transformed_sf = my_tr.fit_transform(sf)
        """
        self.fit(data)
        return self.transform(data)

    @classmethod
    def _get_queryable_methods(cls):
        return {'transform': {}}

    def _get_instance_and_data(self):
        raise NotImplementedError

# Base class for Models written in C++ using the SDK.
# ---------------------------------------------------
class Transformer(TransformerBase):

    _fit_examples_doc = '''
    '''
    _fit_transform_examples_doc = '''
    '''
    _transform_examples_doc = '''
    '''

    def __init__(self, model_proxy = None, _class = None):
        self.__proxy__ = model_proxy
        if _class:
            self.__class__ = _class

    def fit(self, data):
        """
        Fit a transformer using the SFrame `data`.

        Parameters
        ----------
        data : SFrame
            The data used to fit the transformer.

        Returns
        -------
        self (A fitted version of the object)

        See Also
        --------
        transform, fit_transform

        Examples
        --------
        .. sourcecode:: python

        {examples}
        """

        _raise_error_if_not_sframe(data, "data")
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.fit')
        self.__proxy__.fit(data)
        return self

    def transform(self, data):
        """
        Transform the SFrame `data` using a fitted model.

        Parameters
        ----------
        data : SFrame
            The data  to be transformed.

        Returns
        -------
        A transformed SFrame.

        See Also
        --------
        fit, fit_transform

        Examples
        --------
        .. sourcecode:: python

        {examples}

        """
        _raise_error_if_not_sframe(data, "data")
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.transform')
        return self.__proxy__.transform(data)

    def fit_transform(self, data):
        """
        First fit a transformer using the SFrame `data` and then return a
        transformed version of `data`.

        Parameters
        ----------
        data : SFrame
            The data used to fit the transformer. The same data is then also
            transformed.

        Returns
        -------
        Transformed SFrame.

        See Also
        --------
        fit, transform

        Notes
        ------
        - Fit transform modifies self.

        Examples
        --------
        .. sourcecode:: python

        {examples}
        """
        _raise_error_if_not_sframe(data, "data")
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.fit_transform')
        return self.__proxy__.fit_transform(data)

    def _get_wrapper(self):
        _class = self.__class__
        proxy_wrapper = self.__proxy__._get_wrapper()
        def model_wrapper(unity_proxy):
            model_proxy = proxy_wrapper(unity_proxy)
            return Transformer(model_proxy, _class)
        return model_wrapper

    def save(self, location):
        """
        Save the transformer into a GraphLab archive. The object is saved as a
        directory which can then be loaded using the
        :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        location : string
            Target destination for the model. Can be a local path or remote
            URL.

        See Also
        ----------
        graphlab.load_model

        Examples
        ----------
        .. sourcecode:: python

            >>> model.save('my_model_file')
            >>> loaded_model = gl.load_model('my_model_file')

        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.save')
        return glconnect.get_unity().save_model(self.__proxy__,
                             _make_internal_url(location), self._get_wrapper())

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and train the
        transformer.

        Returns
        -------
        out : dict
            Dictionary with options used to define and train the transformer.

        See Also
        ---------
        get, list_fields

        Examples
        ----------
        >>> options = model.get_current_options()
        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + \
                                                    '.get_current_options')
        return self.__proxy__.get_current_options()

    def list_fields(self):
        """
        List of fields stored in the model. Each of these fields can be queried
        using the ``get(field)`` function or ``m[field]``.

        Returns
        -------
        out : list[str]
            A list of fields that can be queried using the ``get`` method.

        See Also
        ---------
        get

        Examples
        --------
        >>> fields = m.list_fields()
        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + \
                                                    '.list_fields')
        return self.__proxy__.list_fields()

    def get(self, field):
        """Return the value for the queried field.

        Each of these fields can be queried in one of two ways:

        >>> out = m['field']
        >>> out = m.get('field')  # equivalent to previous line

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        See Also
        ---------
        list_fields

        Returns
        -------
        out : value
            The current value of the requested field.
        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.get')
        if field in self.list_fields():
            return self.__proxy__.get(field)
        else:
            raise KeyError('Field \"%s\" not in model. Available fields are '
                         '%s.' % (field, ', '.join(self.list_fields())))

    def __getitem__(self, key):
        return self.get(key)

    @classmethod
    def _is_gl_pickle_safe(cls):
        """
        Return True if the model is GLPickle safe i.e if the model does not
        contain elements that are written using Python + GraphLab objects.
        """
        return False

    @classmethod
    def _get_queryable_methods(cls):
        return {'transform': {'data': 'sframe'}}

class _SampleTransformer(Transformer):

    get_default_options = staticmethod(_get_default_options_wrapper(
        '_SampleTransformer', '_SampleTransformer', '_SampleTransformer', True))

    def __init__(self, features=None, constant=0.5):

        # Set up options
        opts = {}
        opts['features'] = features
        opts['constant'] = constant

        # Initialize object
        proxy = _gl.extensions._SampleTransformer()
        proxy.init_transformer(opts)
        super(_SampleTransformer, self).__init__(proxy, self.__class__)

    def _get_summary_struct(self):
        """
        Returns a structured description of the model, including (where
        relevant) the schema of the training data, description of the training
        data, training statistics, and model hyperparameters.

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
        section = []
        section_titles = ['Attributes']
        for f in self.list_fields():
            section.append( ("%s" % f,"%s"% f) )

        return ([section], section_titles)

    def __repr__(self):
        (section, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, section, section_titles, width=30)

