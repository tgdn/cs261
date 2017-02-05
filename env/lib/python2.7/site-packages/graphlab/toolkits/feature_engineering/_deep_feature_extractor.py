import graphlab as _gl
import copy as _copy

# GLC Internal utils.
from graphlab.data_structures.image import Image as _Image
from graphlab.data_structures.sframe import SFrame as _SFrame
from graphlab.util import _raise_error_if_not_of_type
from graphlab.toolkits._internal_utils import _toolkit_repr_print
from graphlab.toolkits._internal_utils import _check_categorical_option_type
from graphlab.toolkits._internal_utils import _raise_error_if_column_exists
import graphlab as _gl

# GLC Utils.
import graphlab.connect as _mt
from graphlab.toolkits._main import ToolkitError
from graphlab.toolkits.classifier.neuralnet_classifier \
                        import NeuralNetClassifier as _NeuralNetClassifier

# Feature engineering utils.
from ._feature_engineering import TransformerBase as _TransformerBase


def _get_default_options(output_type = 'sframe'):
    """
    Return information about the default options.

    Parameters
    ----------
    output_type : str, optional

        The output can be of the following types.

        - `sframe`: A table description each option used in the model.
        - `json`: A list of option dictionaries.

        | Each dictionary/row in the JSON/SFrame object describes the
          following parameters of the given model.

        +------------------+-------------------------------------------------------+
        |      Name        |                  Description                          |
        +==================+=======================================================+
        | name             | Name of the option used in the model.                 |
        +------------------+---------+---------------------------------------------+
        | description      | A detailed description of the option used.            |
        +------------------+-------------------------------------------------------+
        | type             | Option type.                                          |
        +------------------+-------------------------------------------------------+
        | default_value    | The default value for the option.                     |
        +------------------+-------------------------------------------------------+
        | possible_values  | List of acceptable values (CATEGORICAL only)          |
        +------------------+-------------------------------------------------------+
        | lower_bound      | Smallest acceptable value for this option (REAL only) |
        +------------------+-------------------------------------------------------+
        | upper_bound      | Largest acceptable value for this option (REAL only)  |
        +------------------+-------------------------------------------------------+

    Returns
    -------
    out : JSON/SFrame
        Each row in the output SFrames correspond to a parameter, and includes
        columns for default values, lower and upper bounds, description ,and
        type.
    """

    _check_categorical_option_type('output_type', output_type,
                                    ['json', 'sframe'])
    import graphlab as _gl
    sf = _gl.SFrame({
        'name': ['model'],
        'default_value': ['auto'],
        'lower_bound' : [None],
        'upper_bound' : [None],
        'parameter_type' : ['Model or String'],
        'possible_values' : [None],
    })
    if output_type == "sframe":
        return sf
    else:
        return [row for row in sf]

class DeepFeatureExtractor(_TransformerBase):
    """

    Takes an input dataset, propagates each example through the network,
    and returns an SArray of dense feature vectors, each of which is the
    concatenation of all the hidden unit values at layer[layer_id]. These
    feature vectors can be used as input to train another classifier such as a
    :py:class:`~graphlab.logistic_classifier.LogisticClassifier`,
    an :py:class:`~graphlab.svm_classifier.SVMClassifier`, another
    :py:class:`~graphlab.neuralnet_classifier.NeuralNetClassifier`,
    or a :py:class:`~graphlab.boosted_trees_classifier.BoostedTreesClassifier`.

    A pre-trained model for ImageNet, as described by Alex Krizhevsky et. al.
    is avaliable for use at
    "https://static.turi.com/products/graphlab-create/resources/models/python2.7/imagenet_model_iter45".

    Parameters
    ----------
    features : str, list[str]
        Name of feature column to be transformed.

    model: 'auto' | A model of type NeuralNetClassifier
        The model to extract features from. By default ('auto'), we chose an
        appropriate model from our batch of pre-trained models.

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

    References
    ----------
    - Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
      classification with deep convolutional neural networks." Advances in
      neural information processing systems. 2012.

    Examples
    --------
    """
    _DEEP_FEATURE_EXTRACTOR_VERSION = 1
    get_default_options = staticmethod(_get_default_options)

    def __init__(self, features, model = 'auto', output_column_prefix=None):
        """
        Parameters
        ----------
        """
        _raise_error_if_not_of_type(features, [str, list, type(None)])
        _raise_error_if_not_of_type(model, [str, _NeuralNetClassifier])
        _raise_error_if_not_of_type(output_column_prefix, [str, type(None)])

        if isinstance(features, str):
            features = [features]

        # Set the model.
        self._state = {}
        self._state["features"] = features
        if not output_column_prefix:
            output_column_prefix = "deep_features"
        self._state["output_column_prefix"] = output_column_prefix

        self._state['model'] = model
        if self._state["model"] == 'auto':
            model_path = \
    "https://static.turi.com/products/graphlab-create/resources/models/python2.7/imagenet_model_iter45"
            import graphlab as gl
            self._state['model'] = gl.load_model(model_path)
        if type(self._state['model']) is not _NeuralNetClassifier:
            raise ValueError("Model parameters must be of type NeuralNetClassifier " +
                "or string literal 'auto'")


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

        fields = [
            ("Feature(s)", "features"),
            ("Output column prefix", 'output_column_prefix'),
        ]
        section_titles = ['Model fields']
        return ([fields], section_titles)

    def __repr__(self):
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=20)

    def fit(self, data):
        """
        Fits a transformer using the SFrame `data`. The `fit` phase does not
        train a deep learning model, it only checks that the trained model
        is comptable with the data provided. If the `auto` model is chosen, then
        the fit phase choses the right model to extract features from.

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

        # Create data.
        >>> import graphlab as gl

        # Import data from MNIST
        >>> data = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')

        # Create a DeepFeatureExtractorObject
        >>> extractor = gl.feature_engineering.DeepFeatureExtractor(features = 'image')

        # Fit the encoder for a given dataset.
        >>> extractor = extractor.fit(data)

        # Return the model used for the deep feature extraction.
        >>> extractor['model']
        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.fit')

        # Check that the column is in the SFrame.
        _raise_error_if_not_of_type(data, [_SFrame])

        for feature in self._state["features"]:
            _raise_error_if_column_exists(data, feature)
            if data[feature].dtype() != _Image:
                raise ToolkitError(
                    "Feature `%s` must be of type Image." % feature)

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

            # Import data from MNIST
            >>> data = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')

            # Create a DeepFeatureExtractorObject
            >>> extractor = gl.feature_engineering.DeepFeatureExtractor(features='image')

            # Fit the extractor for a given dataset.
            >>> data = extractor.fit_transform(data)
        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.transform')
        transformed_data = _copy.copy(data)
        for feature in self._state["features"]:
            transformed_name = '{}.{}'.format(self._state["output_column_prefix"], feature)
            image = _gl.SFrame({'image': data[feature]})
            transformed_data[transformed_name] =\
                                self._state["model"].extract_features(image)
        return transformed_data


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
        out : SFrame.
            The transformed SFrame.

        See Also
        --------
        fit, fit_transform

        Notes
        ------
        Fit transform modifies self.

        Examples
        --------
        .. sourcecode:: python

            >>> import graphlab as gl

            # Import data from MNIST
            >>> data = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')

            # Create a DeepFeatureExtractorObject
            >>> extractor = gl.feature_engineering.DeepFeatureExtractor(
            ...                                         features = 'image')

            # Fit the extractor for a given dataset.
            >>> extractor = extractor.fit(data)

            # Fit the extractor for a given dataset.
            >>> data_with_features = extractor.transform(data)
        """
        self.fit(data)
        return self.transform(data)

    def get(self, field):
        """
        Return the value contained in the model's ``field``.

        Parameters
        ----------
        field : string
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
            raise ValueError("There is no model field called {}.".format(field))

    def list_fields(self):
        """
        List of fields stored in the model. Each of these fields can be queried
        using the ``get(field)`` function or ``m[field]``.

        Returns
        -------
        out : list[str]
            A list of fields that can be queried using the ``get`` method.

        Examples
        --------
        >>> fields = m.list_fields()
        """
        return list(self._state.keys())

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
        return {o: self._state[o] for o in ["model"]}

    def _get_version(self):
        return self._DEEP_FEATURE_EXTRACTOR_VERSION

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
        obj = unpickler.load()
        if version == 0:
            model = obj._state['model']

            # Features was just str, not list
            features = [obj._state['features']]
            # Attribute output_column_prefix did not exist. Use the old
            # output_column_name instead.
            output_column_name = obj._state['output_column_name']
            output_column_prefix = output_column_name
        else:
            model = obj._state['model']
            features = obj._state['features']
            output_column_prefix = obj._state['output_column_prefix']
        return  DeepFeatureExtractor(features=features,
                    model=model,
                    output_column_prefix=output_column_prefix)

    @classmethod
    def _get_instance_and_data(cls):
        from PIL import Image as _PIL_Image
        import random
        _format = {'JPG': 0, 'PNG': 1, 'RAW': 2, 'UNDEFINED': 3}
        # Note: This needs to be added to the OSS repo as an exposed function.
        def from_pil_image(pil_img):
            height = pil_img.size[1]
            width = pil_img.size[0]
            if pil_img.mode == 'L':
                image_data = bytearray([z for z in pil_img.getdata()])
                channels = 1
            elif pil_img.mode == 'RGB':
                image_data = bytearray([z for l in pil_img.getdata() for z in l ])
                channels = 3
            else:
                image_data = bytearray([z for l in pil_img.getdata() for z in l])
                channels = 4
            format_enum = _format['RAW']
            image_data_size = len(image_data)
            img = _gl.Image(_image_data=image_data,
                           _width=width, _height=height,
                           _channels=channels,
                           _format_enum=format_enum,
                           _image_data_size=image_data_size)
            return img

        num_examples = 100
        dims = (28,28)
        images = []
        for i in range(num_examples):
            def rand_image():
                return [random.randint(0,255)] * (28*28)
            pil_img = _PIL_Image.new('RGB', dims)
            pil_img.putdata(list(zip(rand_image(), rand_image(), rand_image())))
            images.append(from_pil_image(pil_img))
            random_labels = random.randint(0,1)

        data = _gl.SFrame({'image': _gl.SArray(images)})
        data['label'] = random_labels
        nn_model = _gl.neuralnet_classifier.create(data, 'label')
        data.remove_column('label')
        extractor = _gl.feature_engineering.DeepFeatureExtractor(
                            features = ['image'], model = nn_model)
        extractor = extractor.fit(data)
        return extractor, data
