"""
The similarity search toolkit searches a reference collection of raw data
objects, such as images, for items that are similar to a query. This toolkit
encapsulates the whole data processing pipeline that typically accompanies this
process: detection of input data types, extraction of numeric features, and a
nearest neighbors search. For advanced users, each phase of this process can be
customized, or the toolkit can choose good default options automatically.
"""

from graphlab.data_structures.sframe import SFrame as _SFrame
from graphlab.data_structures.image import Image as _Image
from graphlab.toolkits._model import CustomModel as _CustomModel
from graphlab.toolkits._main import ToolkitError as _ToolkitError
import graphlab as _gl
import graphlab.toolkits._internal_utils as _tkutl
import graphlab.connect as _mt
from graphlab.util import _raise_error_if_not_of_type
from graphlab.toolkits._internal_utils import _raise_error_if_column_exists
import time as _time


def get_default_options():
    """
    Return default options information for the similarity search toolkit.

    Returns
    -------
    out : SFrame
        Each row in the output SFrames correspond to a parameter, and
        includes columns for default values, lower and upper bounds,
        description, and type.
    """
    out = _SFrame({'name': ['method', 'feature_model', 'verbose'],
                  'default_value' : ['lsh', 'auto', 'True'],
                  'lower_bound': [None, None, 0],
                  'upper_bound': [None, None, 1],
                  'description': ['Method for searching reference data',
                                  'Trained model for extracting features from raw data objects',
                                  'Whether progress output is printed'],
                  'parameter_type': ['string', 'model', 'boolean']})

    return out


def create(data, row_label=None, features=None, feature_model='auto',
           method='lsh', verbose=True):
    """
    Create a similarity search model, which can be used to quickly retrieve
    items similar to a query observation. In the case of images, this model
    automatically performs the appropriate feature engineering steps. NOTE:
    If you are using a CPU for the creation step with feature_model='auto',
    creation time may take a while. This is because extracting features for
    images on a CPU is expensive. With a GPU, one can expect large speedups.

    Parameters
    ----------
    dataset : SFrame
        The SFrame that represents the training data for the model, including at
        least one column of images.

    row_label : str, optional
        Name of the SFrame column with row id's. If 'row_label' is not
        specified, row numbers are used to identify reference dataset rows when
        the model is queried.

    features : str, optional
        The name of an image column in the input 'dataset' SFrame.

    feature_model : 'auto' | A model of type NeuralNetClassifier, optional
        A trained model for extracting features from raw data objects. By
        default ('auto'), we choose an appropriate model from our set of
        pre-trained models. See
        :class:`~graphlab.toolkits.feature_engineering.DeepFeatureExtractor` for
        more information.

    method : {'lsh', 'brute_force'}, optional
        The method used for nearest neighbor search. The 'lsh' option uses
        locality-sensitive hashing to find approximate results more quickly.

    verbose : bool, optional
        If True, print verbose output during model creation.

    Returns
    -------
    out : SimilaritySearchModel

    See Also
    --------
    SimilaritySearchModel
    graphlab.toolkits.nearest_neighbors
    graphlab.toolkits.feature_engineering

    Notes
    -----
    The similarity search toolkit currently uses cosine distance to evaluate the
    similarity between each query and candidate results.

    Examples
    --------
    First, split data into reference and query.

    >>> import graphlab as gl

    >>> data = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')
    >>> reference, query = data.random_split(0.8)

    Build neuralnet feature extractor for images:

    >>> nn_model = gl.neuralnet_classifier.create(reference, target='label')

    Construct SimilaritySearchModel:

    >>> model = gl.similarity_search.create(reference, features= 'image',
    ...                                     feature_model=nn_model)

    Find the most similar items in the reference set for each item in the query
    set:

    >>> model.search(query)
    """

    _mt._get_metric_tracker().track(__name__ + '.create')

    _raise_error_if_not_of_type(data, [_SFrame])
    _raise_error_if_not_of_type(features, [str])
    _raise_error_if_column_exists(data, features)

    if data[features].dtype() != _Image:
        raise _ToolkitError("Feature `%s` must be of type Image" \
                % features)

    return SimilaritySearchModel(data, row_label=row_label, feature=features,
            feature_model=feature_model, method=method, verbose=verbose)


class SimilaritySearchModel(_CustomModel):
    """
    The similarity search toolkit searches a reference collection of raw data
    objects, such as images, for items that are similar to a query. This toolkit
    encapsulates the whole data processing pipeline that typically accompanies
    this process: detection of input data types, extraction of numeric features,
    and a nearest neighbors search. For advanced users, each phase of this
    process can be customized, or the toolkit can choose good default options
    automatically.

    This model should not be constructed directly. Instead, use
    :func:`graphlab.data_matching.similarity_search.create` to create an
    instance of this model.
    """

    _SIMILARITY_SEARCH_VERSION = 1
    def __init__(self, data, row_label=None, feature=None, feature_model='auto',
                 method='brute_force', verbose=False):

        start_time = _time.time()

        self._state = {'row_label': row_label,
                       'method': method,
                       'verbose': verbose,
                       'features': feature,
                       'num_examples': data.num_rows()}

        if row_label is not None:
            data_subset = data[[feature, row_label]]
        else:
            data_subset = data[[feature]]

        self._feature_type = data_subset[feature].dtype()

        if data_subset[feature].dtype() == _Image:
            prefix = 'extracted'
            extractor = _gl.feature_engineering.DeepFeatureExtractor(
                    features=feature, output_column_prefix=prefix,
                    model=feature_model)
            self._state['output_column_name'] = prefix + '.' + feature
            self._state['feature_model'] = extractor['model']
            self._extractor = extractor.fit(data_subset)
            self._data = self._extractor.transform(data_subset)
        else:
            raise _ToolkitError('Feature type not supported.')


        if method == 'brute_force':
            self._neighbors_model = _gl.toolkits.nearest_neighbors.create(
                self._data, label=row_label,
                features=[self._state['output_column_name']],
                distance='cosine', method='brute_force', verbose=verbose)

        elif method == 'lsh':
            num_tables = 20
            num_projections_per_table = 16
            self._neighbors_model = _gl.toolkits.nearest_neighbors.create(
                self._data, label=row_label,
                features=[self._state['output_column_name']],
                distance='cosine', method = 'lsh',
                num_tables=num_tables,
                num_projections_per_table=num_projections_per_table,
                verbose=verbose)

        else:
            raise _ToolkitError('Unsupported Method %s' % method)

        self._state['training_time'] = _time.time() - start_time


    def save(self, location, save_untransformed=False):
        """
        Save the model. The model is saved as a directory which can then be
        loaded using the :py:func:`~graphlab.load_model` method.

        Parameters
        ----------
        location : string
            Target destination for the model. Can be a local path or remote URL.

        save_untransformed: bool
            Whether to save untransformed data (e.g. images) in the 'data' field. Images may take up quite a lot of space, and it may only be necessary to keep the internal representation (extracted features) of those images. Default is false.

        See Also
        ----------
        graphlab.load_model

        Examples
        ----------
        >>> model.save('my_model_file')
        >>> loaded_model = gl.load_model('my_model_file')

        """

        if not save_untransformed:
            temp_untransformed = self._data[self._state['features']]
            del self._data[self._state['features']]
            super(SimilaritySearchModel,self).save(location)
            self._data[self._state['features']] = temp_untransformed
        else:
            super(SimilaritySearchModel,self).save(location)

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An function to load an object with a specific version of the class.

        Parameters
        ----------
        pickler : file
            A GLUnpickler file handle.

        version : int
            A version number as maintained by the class writer.
        """
        model = unpickler.load()
        if version == 0:
            feature = model._state['features']
            model._state['output_column_name'] = 'extracted.' + feature
        return model

    def list_fields(self):
        """
        List the model's queryable fields.

        Returns
        -------
        out : list
            Each element in the returned list can be queried with the ``get``
            method.
        """
        return list(self._state.keys())

    def get(self, field):
        """
        Return the value contained in the model's ``field``.

        The list of all queryable fields is detailed below, and can be obtained
        with the ``list_fields`` method.

        +-----------------------+----------------------------------------------+
        |      Field            | Description                                  |
        +=======================+==============================================+
        | feature_model         | Model for extracting features from raw data. |
        +-----------------------+----------------------------------------------+
        | features              | Name of the feature column in the input data.|
        +-----------------------+----------------------------------------------+
        | method                | Method for searching the reference data.     |
        +-----------------------+----------------------------------------------+
        | num_examples          | Number of reference data objects.            |
        +-----------------------+----------------------------------------------+
        | row_label             | Name of the row ID column.                   |
        +-----------------------+----------------------------------------------+
        | training_time         | Time to create the model.                    |
        +-----------------------+----------------------------------------------+
        | verbose               | Whether model creation progress is printed.  |
        +-----------------------+----------------------------------------------+

        Parameters
        ----------
        field : str
            Name of the field to be retrieved.

        Returns
        -------
        out
            Value of the requested field.

        See Also
        --------
        list_fields
        """
        try:
            return self._state[field]
        except:
            raise ValueError("There is no model field called {}".format(field))

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current model.

        Returns
        -------
        out : dict
            Dictionary of option and values used to train the current instance
            of the NearestNeighborDeduplication.

        See Also
        --------
        get_default_options, list_fields, get
        """
        return {k: self._state[k] for k in get_default_options()['name']}

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the NearestNeighborsModel.
        """
        return self.__repr__()

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

        model_fields = [
            ("Number of reference examples", 'num_examples')]

        training_fields = [
            ("Method", 'method'),
            ("Total training time (seconds)", 'training_time')]

        sections = [model_fields, training_fields]
        section_titles = ['Schema', 'Training']

        return (sections, section_titles)

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """

        (sections, section_titles) = self._get_summary_struct()
        return _tkutl._toolkit_repr_print(self, sections, section_titles, width=30)

    def search(self, data, row_label=None, k=5):
        """
        Search for the nearest neighbors from the reference set for each element
        of the query set. The query SFrame must include columns with the same
        names as the row_label and feature columns used to create the
        SimilaritySearchModel.

        Parameters
        ----------
        data : SFrame
            Query data. Must contain columns with the same names and types as
            the features used to train the model. Additional columns are
            allowed, but ignored.

        row_label : string, optional
            Name of the query SFrame column with row id's. If 'row_label' is not
            specified, row numbers are used to identify query dataset rows in
            the output SFrame.

        k : int, optional
            Number of nearest neighbors to return from the reference set for
            each query observation. The default is 5 neighbors.

        Returns
        -------
        out
            A SFrame that contains all the nearest neighbors.

        Examples
        --------
        First, split data into reference and query:

        >>> import graphlab as gl
        >>> data = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')
        >>> reference, query = data.random_split(0.8)

        Build a neural net feature extractor for images:

        >>> nn_model = gl.neuralnet_classifier.create(reference, target='label')

        Construct the SimilaritySearchModel:

        >>> model = gl.similarity_search.create(reference, features='image',
        ...                                     feature_model=nn_model)

        Find the most similar items in the reference set for each query:

        >>> model.search(query)
        """

        _raise_error_if_not_of_type(row_label, [str, type(None)])
        feature = self._state['features']
        _raise_error_if_column_exists(data, feature)

        if (data[feature].dtype() != self._feature_type):
            raise ValueError('Feature columns must have same data type in both reference and query set')

        if row_label != None:
            _raise_error_if_column_exists(data, row_label)

        if data[feature].dtype() == _Image:
            transformed_data = self._extractor.transform(data)
        else:
            transformed_data = data
            transformed_data[self._state['output_column_name']] = transformed_data[feature]

        return self._neighbors_model.query(transformed_data, label=row_label, k=k)
