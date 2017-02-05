
import graphlab as _gl
import graphlab.connect as _mt
from graphlab.toolkits.feature_engineering._feature_engineering import Transformer
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _raise_error_if_not_sframe
from graphlab.toolkits._internal_utils import _raise_error_if_sframe_empty
from graphlab.toolkits._internal_utils import _precomputed_field
from . import _internal_utils  # feature engineering utilities


class RandomProjection(Transformer):
    """
    Project high-dimensional numeric features into a low-dimensional subspace.
    This is useful for data visualization and feature engineering, especially
    for tasks that depend on pairwise distances between data points, such as
    clustering and nearest neighbors search.

    This tool uses a *Gaussian random projection* to reduce the dimension of
    dense input data. Each entry of the projection matrix is drawn from a
    standard normal distribution, and the output data is the product of the
    input data and the projection matrix (with a scaling factor). Please see
    the notes and references below for more detail.

    Parameters
    ----------
    features : list[str] or str, optional
        Names of feature columns to transform. If both `features` and
        `excluded_features` are left unspecified, then all columns are used.
        Features must be one of the following types:

        - *numeric*: integer or float values.

        - *array*: list of numeric (integer or float) values. Each element in
          the array is treated as a separate feature in the projection.

        List and dictionary features are *not* currently allowed, but will be
        enabled in future versions of GraphLab Create.

    excluded_features : list[str] or str, optional
        Names of feature columns in the input dataset to be ignored. Either
        `excluded_features` or `features` can be passed, but not both.

    embedding_dimension : int, optional
        Number of features in the output low-dimensional space. By default,
        this is set to 2.

    output_column_name : str, optional
        The projected data is in a single column in the SFrame returned by the
        :func:`transform` and :func:`fit_transform` methods. This parameter is
        the name of that column.

    random_seed : int, optional
        Seed for randomly generating entries in the projection matrix. If two
        :class:`RandomProjection` instances have the same random seed, they
        yield the same output data point for a given input data point.

    Returns
    -------
    out : RandomProjection
        A :class:`RandomProjection` feature transformer object, initialized
        with the defined parameters.

    Notes
    -----
    - If the original dataset is represented by the :math:`n \\times d` matrix
      :math:`X`, then the Gaussian random projection is

      .. math::

        Y = \\frac{1}{\sqrt{k}}XR

      where the output dataset :math:`Y` has :math:`n` rows and :math:`k`
      columns and the random projection matrix :math:`R` has :math:`d` rows and
      :math:`k` columns, with entries drawn from a standard Gaussian
      distribution:

      .. math::

        r_{ij} \sim N(0, 1).

    - The *original dimension* of the projection (also known as the *ambient
      dimension*) must be set by calling the :func:`fit` method of the
      :class:`RandomProjection` object, or calling :func:`fit_transform`, which
      creates and applies a projection matrix all at once.

    References
    ----------
    - Achlioptas, D. (2003). `Database-friendly random projections:
      Johnson-Lindenstrauss with binary coins
      <https://users.soe.ucsc.edu/~optas/papers/jl.pdf>`_. Journal of Computer
      and System Sciences, 66(4), pp. 671-687.

    - Li, P., Hastie, T. J., & Church, K. W. (2006). `Very sparse random
      projections <http://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf>`_.
      Proceedings of the 12th ACM SIGKDD International Conference on Knowledge
      Discovery and Data Mining - KDD '06.

    Examples
    --------
    >>> from graphlab.toolkits.feature_engineering import RandomProjection
    ...
    >>> sf = graphlab.SFrame({'a': [1, 2, 3],
    ...                       'b': [2, 3, 5],
    ...                       'c': [9, 7, 6],
    ...                       'd': [5, 1, 2]})
    >>> projector = graphlab.feature_engineering.create(sf,
    ...                             RandomProjection(features=['a', 'b', 'd']))
    >>> embedded_sf = projector.transform(sf)
    >>> embedded_sf.print_rows()
    +---+-------------------------------+
    | c |       embedded_features       |
    +---+-------------------------------+
    | 9 | [-3.92417400997, 0.1595433... |
    | 7 | [-0.395765702969, 2.083336... |
    | 6 | [-0.820680837147, 3.402468... |
    +---+-------------------------------+
    [3 rows x 2 columns]
    """

    _metric_handle = 'toolkits.feature_engineering.dimension_reduction.random_projection'

    get_default_options = staticmethod(_get_default_options_wrapper(
        unity_server_model_name='_RandomProjection',
        module_name='toolkits.feature_engineering._dimension_reduction',
        python_class_name='RandomProjection', sdk_model=True))

    def __init__(self, features=None, excluded_features=None,
                 embedding_dimension=2, output_column_name='embedded_features',
                 random_seed=None):

        _mt._get_metric_tracker().track(self._metric_handle + '.__init__')

        ## Input validation
        # Process and make a copy of the features, exclude.
        _features, _exclude = _internal_utils.process_features(features,
                                                             excluded_features)

        ## Initialize the C++ RandomProjection object and its Python proxy.
        opts = {'embedding_dimension': embedding_dimension,
                'output_column_name': output_column_name,
                'random_seed': random_seed}

        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features

        proxy = _gl.extensions._RandomProjection()
        proxy.init_transformer(opts)
        super(RandomProjection, self).__init__(proxy, self.__class__)

    def fit(self, data):
        """
        Create a random projection matrix. Input data is needed for this
        function to determine the original dimension and valid feature types
        for the :func:`transform` function, which can be tricky to do
        for complex feature types (e.g. arrays).

        Parameters
        ----------
        data : SFrame
            Data used to create the transformer. This dataset is only used to
            determine the original dimension and to validate the feature types
            of that data that will be passed to :func:`transform`.

        Returns
        -------
        self :
            A fitted version of the :class:`RandomProjection` instance.

        See Also
        --------
        transform, fit_transform

        Notes
        -----
        - Missing values in the input dataset are not typically a problem for
          the :func:`fit` method, but if an array-type column has missing
          values for the first 30 entries, the function cannot determine the
          number of features in that column. In that case, consider passing a
          specific subset of rows that do not have missing values.

        Examples
        --------
        >>> sf = graphlab.SFrame({'a': [1, 2, 3],
        ...                       'b': [2, 3, 5],
        ...                       'c': [9, 7, 6],
        ...                       'd': [5, 1, 2]})
        >>> projector = graphlab.feature_engineering.RandomProjection(
        ...                                          features=['a', 'b', 'd'])
        >>> projector = projector.fit(sf)
        >>> projector.summary()
        Class                         : RandomProjection
        ...
        Embedding dimension           : 2
        Original dimension            : 3
        Features                      : ['a', 'b', 'd']
        Excluded features             : None
        Output column name            : embedded_features
        Random seed                   : 1456791539
        Has been fitted               : 1
        """
        _raise_error_if_not_sframe(data, "data")
        _raise_error_if_sframe_empty(data, "data")

        self.__proxy__.fit(data)
        return self

    def transform(self, data):
        """
        Transform the input dataset using a fitted :class:`RandomProjection`
        transformer.

        Parameters
        ----------
        data : SFrame
            The data to be transformed.

        Returns
        -------
        out : SFrame
            A transformed SFrame. Columns not used by the transformer remain
            unchanged in the output SFrame. The embedded data (i.e. projected
            data) is contained in a single array-typed column of the output
            SFrame, named according to the `output_column_name` parameter in
            the model constructor.

        See Also
        --------
        fit, fit_transform

        Notes
        -----
        - The data used for the :func:`transform` function cannot have missing
          data, because the dimension of affected rows would not match the
          dimension of the projection matrix, which has already been created.
          Please use the :func:`graphlab.SFrame.dropna` or
          :func:`graphlab.SFrame.fillna` methods to address missing values.

        Examples
        --------
        >>> sf = graphlab.SFrame({'a': [1, 2, 3],
        ...                       'b': [2, 3, 5],
        ...                       'c': [9, 7, 6],
        ...                       'd': [5, 1, 2]})
        >>> projector = graphlab.feature_engineering.RandomProjection(
        ...                                          features=['a', 'b', 'd'])
        >>> projector = projector.fit(sf)
        >>> embedded_sf = projector.transform(sf)
        >>> embedded_sf.print_rows()
        +---+-------------------------------+
        | c |       embedded_features       |
        +---+-------------------------------+
        | 9 | [5.52216236465, -7.1836248... |
        | 7 | [4.66171487553, -4.0223500... |
        | 6 | [7.81757565029, -6.7887529... |
        +---+-------------------------------+
        [3 rows x 2 columns]
        """
        _mt._get_metric_tracker().track(self._metric_handle + '.transform')
        _raise_error_if_not_sframe(data, "data")
        _raise_error_if_sframe_empty(data, "data")
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
        out : SFrame
            A transformed SFrame. Columns not used by the transformer remain
            unchanged in the output SFrame. The embedded data (i.e. projected
            data) is contained in a single array-typed column of the output
            SFrame, named according to the `output_column_name` parameter in
            the model constructor.

        See Also
        --------
        fit, transform

        Notes
        ------
        - :func:`fit_transform` modifies the transformer object.

        - The data used for :func:`fit_transform` function cannot have missing
          data, because the dimension of affected rows would not match the
          dimension of the projection matrix, which has already been created.
          Please use the :func:`graphlab.SFrame.dropna` or
          :func:`graphlab.SFrame.fillna` methods to address missing values.

        Examples
        --------
        >>> sf = graphlab.SFrame({'a': [1, 2, 3],
        ...                       'b': [2, 3, 5],
        ...                       'c': [9, 7, 6],
        ...                       'd': [5, 1, 2]})
        >>> projector = graphlab.feature_engineering.RandomProjection(
        ...                                          features=['a', 'b', 'd'])
        >>> embedded_sf = projector.fit_transform(sf)
        >>> embedded_sf.print_rows()
        +---+--------------------------------+
        | c |       embedded_features        |
        +---+--------------------------------+
        | 9 | [3.25867048754, -4.8490175...  |
        | 7 | [0.545408593534, -3.252476...  |
        | 6 | [1.4891247552, -5.52039555979] |
        +---+--------------------------------+
        [3 rows x 2 columns]
        """
        _mt._get_metric_tracker().track(self._metric_handle + '.fit_transform')
        _raise_error_if_not_sframe(data, "data")
        _raise_error_if_sframe_empty(data, "data")
        return self.__proxy__.fit_transform(data)

    def _get_summary_struct(self):
        """
        Return a structured description of the model, including the schema of
        the training data, description of the training data, training
        statistics, and model hyperparameters.
        """

        _features = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('features')))

        _exclude = _precomputed_field(
            _internal_utils.pretty_print_list(self.get('excluded_features')))

        fields = [("Embedding dimension", 'embedding_dimension'),
                  ("Original dimension", 'original_dimension'),
                  ("Features", _features),
                  ("Excluded features", _exclude),
                  ("Output column name", 'output_column_name'),
                  ("Random seed", 'random_seed'),
                  ("Has been fitted", 'is_fitted')]

        section_titles = ["Model fields"]
        return ([fields], section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, 30)

    @classmethod
    def _get_instance_and_data(cls):
        sf = _gl.SFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        projector = \
            _gl.feature_engineering.RandomProjection(features=['a', 'b'])
        return projector.fit(sf), sf

