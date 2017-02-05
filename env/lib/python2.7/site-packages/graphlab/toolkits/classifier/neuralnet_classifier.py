"""
Neuralnet classifier module.
"""
from graphlab.data_structures.image import Image as _Image
import graphlab.connect as _mt
import graphlab.toolkits.deeplearning._main as _deeplearning
import graphlab.toolkits._main as _toolkits_main
import graphlab.toolkits._supervised_learning as _sl
from graphlab.toolkits._supervised_learning import Classifier as _Classifier
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits._internal_utils import _raise_error_if_not_sframe, \
                                            _map_unity_proxy_to_object, \
                                            _toolkit_repr_print, \
                                            _numeric_param_check_range
from graphlab.toolkits._model_workflow import _collect_model_workflow
from graphlab.util import cloudpickle as _cloudpickle

import logging as _logging
from copy import copy as _copy
import six as _six


get_default_options = _get_default_options_wrapper(
                          'neuralnet_classifier_v2',
                          'neuralnet_classifier',
                          'NeuralNetClassifier')

_context_doc_string = '''
>>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train')
>>> training_data, validation_data = data.random_split(0.8)
>>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')
>>> m = graphlab.neuralnet_classifier.create(training_data,
...                                          target='label',
...                                          network=net,
...                                          max_iterations=3)
'''


class NeuralNetClassifier(_Classifier):
    """
    Neural Network is one of the classical models in artificial intelligence and
    machine learning, and has recently achieved great success in computer vision
    tasks such as object recognition.

    This model cannot be constructed directly.  Instead, use
    :func:`graphlab.neuralnet_classifier.create` to create an instance of
    this model.
    Additional details on parameter options and code samples are available in
    the documentation for the create function.

    See Also
    --------
    create
    graphlab.deeplearning.NeuralNet
    graphlab.deeplearning.create

    Examples
    --------
    >>> # Load the data (From an S3 bucket)
    >>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')
    >>> test_data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/test')
    ...
    >>> # Create a neural network classifier with the MNIST architecture
    >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')
    >>> model = graphlab.neuralnet_classifier.create(data, target='label',
    ...                                              network=net)
    ...
    >>> # Classify test data and evaluate the model
    >>> pred = model.classify(test_data)
    >>> results = model.evaluate(test_data)
    """

    def __init__(self, model_proxy):
        """__init__(self)"""
        self.__proxy__ = model_proxy
        self.__name__ = "neuralnet_classifier_v2"

    def _get_wrapper(self):
        def model_wrapper(model_proxy):
            return NeuralNetClassifier(model_proxy)
        return model_wrapper

    def get(self, field):
        """
        Get the value of a given field. The list of all queryable fields can
        be obtained programmatically using the :func:`~graphlab.neuralnet_classifier.NeuralNetClassifier.list_fields` method.

        Parameters
        ----------
        field: string
            The string of the field to be queried.

        Returns
        -------
        out :
            Value of queried field

        Examples
        --------

        >>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')
        >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')
        >>> m = graphlab.neuralnet_classifier.create(data, target='label',
        ...                                          network = net,
        ...                                          metric=['accuracy', 'recall@2'],
        ...                                          max_iterations=1)
        >>> m.list_fields()
        ['batch_size',
         'bias_l2_regularization',
         'device',
         'features',
         'init_random',
         'l2_regularization',
         'learning_rate',
         'learning_rate_alpha',
         'learning_rate_gamma',
         'learning_rate_schedule',
         'learning_rate_start_epoch',
         'learning_rate_step',
         'max_iterations',
         'metric',
         'min_learning_rate',
         'model_checkpoint_interval',
         'model_checkpoint_path',
         'momentum',
         'network',
         'num_examples',
         'num_feature_columns',
         'num_features',
         'num_iterations',
         'random_crop',
         'random_mirror',
         'subtract_mean',
         'target',
         'training_accuracy',
         'training_recall@2',
         'training_time',
         'validation_accuracy',
         'validation_recall@2']
        >>> m.get('num_iterations')
        """
        result = super(NeuralNetClassifier, self).get(field)
        if field == 'network':
            result = _deeplearning.loads(result)
        return result

    @_collect_model_workflow
    def evaluate(self, dataset, metric='auto'):
        """
        Evaluate the model by making predictions of target values and comparing
        these to actual values. Input dataset must be the same size as for the
        training of the model, except for images which are automatically resized.

        Parameters
        ----------
        dataset : SFrame
            Dataset in the same format used for training. The columns names and
            types of the dataset must be the same as that used in training.

        metric : {'auto', 'accuracy', 'recall@1', 'recall@5', ...}, optional
            To evaluate multiple metrics, supply a list of metric names, e.g.
            ['accuracy', 'recall@1', 'recall@5'].

        Returns
        -------
        out : dict
            Dictionary from metric name to value.

        Examples
        --------
        >>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train')
        >>> training_data, validation_data = data.random_split(0.8)
        >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')
        >>> m = graphlab.neuralnet_classifier.create(training_data,
        ...                                          target='label',
        ...                                          network=net,
        ...                                          max_iterations=3)
        ...
        >>> eval_ = m.evaluate(validation_data, metric=['accuracy', 'confusion_matrix'])
        {'accuracy': 0.9624793529510498, 'confusion_matrix':
        +--------------+-----------------+-------+
        | target_label | predicted_label | count |
        +--------------+-----------------+-------+
        |      0       |        0        |  1187 |
        |      2       |        0        |   2   |
        |      3       |        0        |   3   |
        |      4       |        0        |   1   |
        |      5       |        0        |   1   |
        |      6       |        0        |   5   |
        |      7       |        0        |   3   |
        |      8       |        0        |   4   |
        |      9       |        0        |   4   |
        |      1       |        1        |  1296 |
        |     ...      |       ...       |  ...  |
        +--------------+-----------------+-------+
        [77 rows x 3 columns]}

        See which digit is most misclassified:

        >>> cf_mat = eval_['confusion_matrix']
        >>> cf_mat[cf_mat['target_label'] != cf_mat['predicted_label']].groupby(
        ...     'target_label', graphlab.aggregate.SUM('count'))
        +--------------+--------------+
        | target_label | Sum of count |
        +--------------+--------------+
        |      0       |      17      |
        |      3       |      41      |
        |      1       |      29      |
        |      6       |      22      |
        |      2       |      33      |
        |      8       |      92      |
        |      5       |      38      |
        |      4       |      53      |
        |      9       |      46      |
        |      7       |      83      |
        +--------------+--------------+
        [10 rows x 2 columns]
        """
        _mt._get_metric_tracker().track('toolkit.classifier.neuralnet_classifier.evaluate')
        if type(metric) is list:
            metric = ','.join(metric)
        return super(NeuralNetClassifier, self).evaluate(dataset, metric=metric)

    def predict(self, dataset, output_type="class"):
        """
        Return the model predictions for ``dataset``. Input dataset size must be
        the same as for the training of the model, except for images which are
        automatically resized.

        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        output_type : {"class"}, optional
            Choose the return type of the prediction. Available output_types are:

            - `class`: output the class label

        Returns
        -------
        out : SArray
            An SArray with model predictions.

        See Also
        --------
        predict_topk, classify, evaluate

        Examples
        --------
        >>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train')
        >>> training_data, validation_data = data.random_split(0.8)
        >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')
        >>> m = graphlab.neuralnet_classifier.create(training_data,
        ...                                          target='label',
        ...                                          network=net,
        ...                                          max_iterations=3)
        ...
        >>> pred = m.predict(validation_data)
        >>> pred
        dtype: int
        Rows: 12060
        [4, 7, 6, 1, 4, 7, 1, 8, 6, 0, 6, 2, 7, 5, 1, 7, 1, 1, ... ]
        """
        _mt._get_metric_tracker().track('toolkit.classifier.neuralnet_classifier.predict')
        return super(NeuralNetClassifier, self).predict(dataset, output_type=output_type)

    def extract_features(self, dataset, layer_id=None):
        """
        Takes an input dataset, propagates each example through the network,
        and returns an SArray of dense feature vectors, each of which is the concatenation
        of all the hidden unit values at layer[layer_id]. These feature vectors
        can be used as input to train another classifier such as a :py:class:`~graphlab.logistic_classifier.LogisticClassifier`,
        an :py:class:`~graphlab.svm_classifier.SVMClassifier`, another
        :py:class:`~graphlab.neuralnet_classifier.NeuralNetClassifier`, or a :py:class:`~graphlab.boosted_trees_classifier.BoostedTreesClassifier`. Input dataset size must be the same as for the training of the model,
        except for images which are automatically resized.


        We also are releasing a pre-trained model for ImageNet, as described by
        Alex Krizhevsky et. al. It is located at
        https://static.turi.com/products/graphlab-create/resources/models/python2.7/imagenet_model_iter45 .
        Using it requires 256 x 256 x 3 images.
        Please see Examples and References for more.


        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        layer_id : int , optional
            The index of the layer in neuralnet at which the activations are
            taken to be a dense feature vector. Must be a fully-connected layer.
            Default is None, in which case the layer before the connection
            layer to the output is used.


        Returns
        -------
        out : SArray
            An SArray of dtype array.array containing extracted features.

        See Also
        ------------
        graphlab.deeplearning.layers

        References
        ----------
        - Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
          classification with deep convolutional neural networks." Advances in
          neural information processing systems. 2012.

        Examples
        --------
        >>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')
        >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')
        >>> m = graphlab.neuralnet_classifier.create(data,
        ...                                          target='label',
        ...                                          network=net,
        ...                                          max_iterations=3)
        >>> # Now, let's extract features from the last layer
        >>> data['features'] = m.extract_features(data)
        >>> # Now, let's build a new classifier on top of extracted features
        >>> m = graphlab.classifier.create(data,
        ...                                          features = ['features'],
        ...                                          target='label')

        Now, let's see how to load the ImageNet model, and use it for extracting
        features after resizing the data:

        >>> imagenet_model = graphlab.load_model('https://static.turi.com/products/graphlab-create/resources/models/python2.7/imagenet_model_iter45')
        >>> data['image'] = graphlab.image_analysis.resize(data['image'], 256, 256, 3, decode=True)
        >>> data['imagenet_features'] = imagenet_model.extract_features(data)

        """
        _mt._get_metric_tracker().track('toolkit.classifier.neuralnet_classifier.extract_features')
        _raise_error_if_not_sframe(dataset, "dataset")
        options = dict()

        net = self.get('network').layers
        network_size = len(net) - 1
        if layer_id is None:
            if net[network_size]._type == "CONNECTION":
                layer_id = network_size - 1
            else:
                layer_id = network_size - 2
        _numeric_param_check_range("layer_id", layer_id, 0, network_size)

        conv2flat = False
        for i in range(0, layer_id + 1):
            if net[i]._type == "CONNECTION" or net[i]._type == "TRANSITION":
                conv2flat = True

        if conv2flat is not True:
            raise ValueError("Features must be extracted from either a network "
                    "with non-image input or a layer after a FlattenLayer. "
                    "Try extracting features from layer following a FlattenLayer.")

        options.update({'model': self.__proxy__,
                        'model_name': self.__name__,
                        'dataset': dataset,
                        'missing_value_action': "error",
                        'layer_id': layer_id})
        target = _toolkits_main.run('supervised_learning_feature_extraction', options)
        return _map_unity_proxy_to_object(target['extracted'])



    def predict_topk(self, dataset, output_type="probability", k=3):
        """
        Return top-k predictions for the ``dataset``, using the trained model.
        Predictions are returned as an SFrame with three columns: `row_id`,
        `class`, and `probability`,`rank`, or `score`, depending on the ``output_type``
        parameter. Input dataset size must be the same as for training of the
        model, except for images which are automatically resized.

        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        output_type : {'probability', 'rank', 'score'}, optional
            Choose the return type of the prediction:

            - `rank`: outputs rank along with class label.
            - `probability`: outputs learned probability along with class label.
            - `score`: Same as probability

        k : int, optional
            Number of classes to return for each input example.

        Returns
        -------
        out : SFrame
            An SFrame with model predictions.

        See Also
        --------
        predict, classify, evaluate

        Examples
        --------
        >>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train')
        >>> training_data, validation_data = data.random_split(0.8)
        >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')
        >>> m = graphlab.neuralnet_classifier.create(training_data,
        ...                                          target='label',
        ...                                          network=net,
        ...                                          max_iterations=3)
        ...
        >>> pred = m.predict_topk(validation_data, k=3)
        >>> pred
        +--------+-------+-------------------+
        | row_id | class |    probability    |
        +--------+-------+-------------------+
        |   0    |   4   |   0.995623886585  |
        |   0    |   9   |  0.0038311756216  |
        |   0    |   7   | 0.000301006948575 |
        |   1    |   1   |   0.928708016872  |
        |   1    |   3   |  0.0440889261663  |
        |   1    |   2   |  0.0176190119237  |
        |   2    |   3   |   0.996967732906  |
        |   2    |   2   |  0.00151345680933 |
        |   2    |   7   | 0.000637513934635 |
        |   3    |   1   |   0.998070061207  |
        |  ...   |  ...  |        ...        |
        +--------+-------+-------------------+
        [35688 rows x 3 columns]
        """
        _mt._get_metric_tracker().track('toolkit.classifier.neuralnet_classifier.predict_topk')
        _raise_error_if_not_sframe(dataset, "dataset")
        options = dict()
        options.update({'model': self.__proxy__,
                        'model_name': self.__name__,
                        'dataset': dataset,
                        'output_type': output_type,
                        'topk': k,
                        'missing_value_action': 'error'})
        target = _toolkits_main.run('supervised_learning_predict_topk', options)
        return _map_unity_proxy_to_object(target['predicted'])

    def classify(self, dataset):
        """
        Return a classification for each example in the ``dataset``, using the
        trained neural network model. The output SFrame contains predictions as
        both class labels and probabilities that the predicted class is correct.
        Input dataset size must be the same as for the training of the model,
        except for images which are automatically resized.

        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        Returns
        -------
        out : SFrame
            An SFrame with model predictions i.e class labels and probabilities.


        See Also
        ----------
        evaluate, predict, predict_topk

        Examples
        --------
        >>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train')
        >>> training_data, validation_data = data.random_split(0.8)
        >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')
        >>> m = graphlab.neuralnet_classifier.create(training_data,
        ...                                          target='label',
        ...                                          network=net,
        ...                                          max_iterations=3)

        >>> result = m.classify(validation_data)
        >>> result
        +--------+-------+----------------+
        | row_id | class |  probability   |
        +--------+-------+----------------+
        |   0    |   4   | 0.995623886585 |
        |   1    |   1   | 0.928708016872 |
        |   2    |   3   | 0.996967732906 |
        |   3    |   1   | 0.998070061207 |
        |   4    |   7   | 0.999219059944 |
        |   5    |   7   | 0.991823732853 |
        |   6    |   9   | 0.993408679962 |
        |   7    |   9   | 0.924675405025 |
        |   8    |   8   | 0.980929374695 |
        |   9    |   8   | 0.99672973156  |
        |  ...   |  ...  |      ...       |
        +--------+-------+----------------+
        [11896 rows x 3 columns]
        """
        _mt._get_metric_tracker().track(
            'toolkit.classifier.neuralnet_classifier.classify')
        return super(NeuralNetClassifier, self).classify(dataset)

    def _reset_batch_size(self, new_size):
        """
        Reset the batch_size
        """
        import graphlab.extensions as _ext
        if type(new_size) is not int or new_size <= 0:
            raise ValueError('Invalid value for new_size')
        _ext.reset_batch_size(self.__proxy__, new_size)

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out: string
            A description of the model.
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
        data_fields = [
            ("Examples", 'num_examples'),
            ("Features", 'num_features'),
            ("Target column", 'target')]
        training_fields = []

        metric_key = 'metric'
        if not metric_key in self.list_fields():
            metric_key = 'metrics'
        metrics = self.get(metric_key).split(',')

        for m in metrics:
            training_fields.append(('Training %s' % m, 'training_%s' % m))
            training_fields.append(('Validation %s' % m, 'validation_%s' % m))
        training_fields.append(("Training time (sec)", 'training_time'))

        return ([data_fields, training_fields], ['Schema', 'Training Summary'])

    def __repr__(self):
        """
        Print a string description of the model, when the model name is entered
        in the terminal.
        """
        (section, section_titles) = self._get_summary_struct()

        return _toolkit_repr_print(self,section, section_titles)

    @classmethod
    def _get_queryable_methods(cls):
        '''Returns a list of method names that are queryable through Predictive
        Service'''
        return {'predict': {'dataset': 'sframe'},
                'classify': {'dataset': 'sframe'},
                'predict_topk': {'dataset': 'sframe'}
                }


def create(dataset, target,
           features=None,
           max_iterations=10,
           network=None,
           validation_set='auto',
           verbose=True,
           class_weights=None,
           **kwargs):
    """
    Create a :class:`~graphlab.neuralnet_classifier.NeuralNetClassifier` to predict the
    class of data with numerical features or image data features.

    The optional ``network`` parameter accepts a
    :class:`~graphlab.deeplearning.NeuralNet` object, which defines the
    neural network architecture and learning parameters. It is the most
    important parameter in the model learning process; we recommended starting
    with the default architecture returned by
    :func:`deeplearning.create() <graphlab.deeplearning.create>`,
    then tuning it to best fit your task.

    **Multiple GPU Support**

    The creation of neurlanet classifier takes advantage of multiple
    GPU devices. By default, it uses all the GPUs that are detected.
    You can change the default behavior by setting the runtime config
    "GRAPHLAB_NEURALNET_DEFAULT_GPU_DEVICE_ID". For instance,
    the following code sets the default to only use device 0 and 2.

    .. sourcecode:: python

      graphlab.set_runtime_config("GRAPHLAB_NEURALNET_DEFAULT_GPU_DEVICE_ID", "0,2")

    .. note::

      If there is an imbalance in GPU's, where one GPU is slower than another,
      then the faster GPU will end up waiting on the slower GPU.

    Parameters
    ----------
    dataset : SFrame
        A training dataset, containing feature columns and a target column. If
        the feature column is of type graphlab.Image, all images must be of the
        same size.

    target : str
        The name of the column in ``dataset`` that is the prediction target. The
        values in this column represent classes, and must be of integer or string
        type.

    features : list[str], optional
        Column names of the features used to train the model. Each column must
        contain vectors of floats or there can be one column of Image type.
        The default argument is ``None``, which means all columns are used,
        except the target.

    max_iterations : int, optional
        The maximum number of iterations for boosting.

    network : :py:class:`~graphlab.deeplearning.NeuralNet`, optional
        The :py:class:`~graphlab.deeplearning.NeuralNet` object contains model
        learning parameters and definitions for the network architecture. The
        default is ``None``, but we recommend using
        :py:func:`deeplearning.create() <graphlab.deeplearning.create>` to find
        a default structure for the input data. Because this default structure
        may be suboptimal, tuning the
        :py:class:`~graphlab.deeplearning.NeuralNet` is highly recommended.

    validation_set : SFrame, optional
        A dataset for monitoring the models generalization performance.
        For each row of the progress table, the chosen metrics are computed
        for both the provided training dataset and the validation_set. The
        format of this SFrame must be the same as the training set.
        By default this argument is set to 'auto' and a validation set is
        automatically sampled and used for progress printing. If
        validation_set is set to None, then no additional metrics
        are computed. This is computed once per full iteration.
        The default value is 'auto'.

    class_weights : {dict, 'auto'}, optional

        Weights the examples in the training data according to the given class
        weights. If set to 'None', all classes will be weighted equally. The
        `auto` mode set the class weight to be inversely proportional to number
        of examples in the training data with the given class. If providing
        custom class weights, all classes must be present in the dictionary.

    kwargs : dict, optional
        Additional arguments for training the neural network. All of the
        parameters listed below can be stored in the ``params`` attribute of a
        NeuralNet object. If the same parameter is set in both places, the one
        in the create function overrides the one in the NeuralNet object.

            - ``batch_size`` : int, default 100
                The SGD mini batch size. Larger batch_size will improve per
                iteration speed, but costs more (GPU) or CPU memory.
            - ``model_checkpoint_path`` : str, default ""
                If specified, save the model to the given path every n iterations,
                where n is specified by model_checkpoint_interval.
            - ``model_checkpoint_interval`` : int, default 5
                If model_check_point_path is specified,
                save the model to the given path every n iterations.
            - ``mean_image`` : graphlab.image.Image, default None
                If set and subtract_mean is True, use the provided mean image
                to save computation time.
            - ``metric`` : {'accuracy', 'error', 'recall@5', ...} , default auto
                The metric(s) used for evaluating training and validation data.
                To evaluate on multiple metrics, supply a list of metric strings,
                e.g. ['accuracy', 'recall@5'], or use a comma separated string
                e.g. 'accuracy,recall@5'.
            - ``subtract_mean`` : bool, default True
                If true, subtract the mean from each image. Calculate the mean image
                from the training data or use the provided mean_image. Subtracting the mean
                centers the input data, which helps accelarate neural net training.
            - ``random_crop`` : bool, default False
                If true, apply random crop to the input images. The cropped
                image size is defined by the ``input_shape`` parameter below.
                Random cropping helps prevent the model
                from overfitting by augmenting the dataset.
            - ``input_shape`` : str, default None
                A formated string in the form of channels,width,height, e.g "1,28,28"
                or "3,256,256", indicating the shape of the image after random cropping.
                The ``input_shape`` cannot exceed the shape of the original image size.
            - ``random_mirror`` : bool, default False
                If true, apply random mirror to the input images.Random mirroring helps
                prevent the model from overfitting by augmenting the dataset.
            - ``learning_rate`` : float, default 0.001
                The learning_rate for bias and weights.
            - ``momentum`` : float between [0, 1], default 0.9
                The momentum for bias and weights.
            - ``l2_regularization`` : float, default 0.0005
                L2 regularization for weights.
            - ``bias_learning_rate`` : float, default unused
                Specify the learning rate for bias, overriding `learning_rate`.
            - ``bias_momentum`` : float, default unused
                Specify the momentum for bias, overriding `momentum`.
            - ``bias_l2_regularization`` : float, default 0.0
                The L2 regularization for bias.
            - ``learning_rate_schedule`` : {'constant', 'exponential_decay', 'polynomial_decay'}
                Learning rate scheduling algorithm.

                  - `constant`: Use the same learning rate for all iterations
                  - `exponential_decay`: Exponentially decreases the learning rate over
                    iterations. See the notes section for more details.
                  - `polynomial_decay`: Polynomially decreases the learning rate over
                    iterations. See the notes section for more details.

            - ``learning_rate_start_epoch`` : int, default 0
                start learning rate scheduling after epoch
            - ``min_learning_rate`` : float, default 0.00001
                minimum of learning rate
            - ``learning_rate_step`` : int, default 1
                update the learning rate every learning_rate_step number of epochs.
            - ``learning_rate_gamma`` : float, default 0.1.
                learning decay param used in 'exponential_decay'
            - ``learning_rate_alpha`` : float, default 0.5
                learning decay param used in 'polynomial_decay'
            - ``init_random`` : {'gaussian' | 'xavier'}, default 'gaussian'
                The type of initialization for the weights. Either uses
                random gaussian initialization or Xavier initialization.
                See :py:class:`~graphlab.deeplearning.layers.FullConnectionLayer`
                paramters for more information.
            - ``init_sigma`` : float, default 0.01
                The standard deviation of the gaussian distribution weight initializations
                are drawn from.
            - ``init_bias`` : float, default 0.0
                The initial value of the biases.
            - ``divideby`` : float, default 1.0
                The value by which to scale the input data before it is inserted into
                the network.

    Returns
    -------
    out : NeuralNetClassifier

    See Also
    --------
    NeuralNetClassifier, graphlab.deeplearning.NeuralNet

    Notes
    -----
    For exponential_decay, the learning rate decreases exponentially according
    to the following:

    .. math::

        new\_lr = lr * lr\_gamma^{epoch/ lr\_step}

    For polynomial_decay, the learning rate decreases polynomially according to
    the following:

    .. math::

        new\_lr = lr * (1 + (epoch/lr\_step)*lr\_gamma)^{-lr\_alpha}

    References
    ----------
    - LeCun, Yann, et al. `Gradient-based learning applied to document
      recognition. <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>`_
      Proceedings of the IEEE 86.11 (1998): 2278-2324.

    - `The MNIST database <http://yann.lecun.com/exdb/mnist/>`_

    Examples
    --------
    We train a convolutional neural network for digits recognition, using the
    MNIST data. The data has already been downloaded from the MNIST database,
    and saved as SFrames in the Turi's public S3 bucket.

    >>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train')
    >>> test_data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/test')
    >>> training_data, validation_data = data.random_split(0.8)

    Resize all the images to the same size, since neural nets have fixed
    input size.

    >>> training_data['image'] = graphlab.image_analysis.resize(training_data['image'], 28, 28, 1, decode=True)
    >>> validation_data['image'] = graphlab.image_analysis.resize(validation_data['image'], 28, 28, 1, decode=True)
    >>> test_data['image'] = graphlab.image_analysis.resize(test_data['image'], 28, 28, 1, decode=True)

    Use the built-in NeuralNet architecture for MNIST (a one layer convolutional
    neural network):

    >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')

    Layers of the neural network:

    >>> net.layers
    layer[0]: ConvolutionLayer
      padding = 1
      stride = 2
      random_type = xavier
      num_channels = 32
      kernel_size = 3
    layer[1]: MaxPoolingLayer
      stride = 2
      kernel_size = 3
    layer[2]: FlattenLayer
    layer[3]: DropoutLayer
      threshold = 0.5
    layer[4]: FullConnectionLayer
      init_sigma = 0.01
      num_hidden_units = 100
    layer[5]: SigmoidLayer
    layer[6]: FullConnectionLayer
      init_sigma = 0.01
      num_hidden_units = 10
    layer[7]: SoftmaxLayer

    Parameters of the neural network:

    >>> net.params
    {'batch_size': 100,
     'data_shape': '1,28,28',
     'divideby': 255,
     'init_random': 'gaussian',
     'l2_regularization': 0.0,
     'learning_rate': 0.1,
     'momentum': 0.9}

    Train a NeuralNetClassifier using the specified network:

    >>> m = graphlab.neuralnet_classifier.create(training_data, target='label',
    ...                                          network = net,
    ...                                          validation_set=validation_data,
    ...                                          metric=['accuracy', 'recall@2'],
    ...                                          max_iterations=3)

    Classify the test data, and output the most likely class label.'probability'
    corresponds to the probability that the input belongs to that class:

    >>> pred = m.classify(test_data)
    >>> pred
    +--------+-------+----------------+
    | row_id | class |  probability   |
    +--------+-------+----------------+
    |   0    |   0   | 0.998417854309 |
    |   1    |   0   | 0.999230742455 |
    |   2    |   0   | 0.999326109886 |
    |   3    |   0   | 0.997855246067 |
    |   4    |   0   | 0.997171103954 |
    |   5    |   0   | 0.996235311031 |
    |   6    |   0   | 0.999143242836 |
    |   7    |   0   | 0.999519705772 |
    |   8    |   0   | 0.999182283878 |
    |   9    |   0   | 0.999905228615 |
    |  ...   |  ...  |      ...       |
    +--------+-------+----------------+
    [10000 rows x 3 columns]

    Predict the top 2 most likely digits:

    >>> pred_top2 = m.predict_topk(test_data, k=2)
    >>> pred_top2
    +--------+-------+-------------------+
    | row_id | class |    probability    |
    +--------+-------+-------------------+
    |   0    |   0   |   0.998417854309  |
    |   0    |   6   | 0.000686840794515 |
    |   1    |   0   |   0.999230742455  |
    |   1    |   2   | 0.000284609268419 |
    |   2    |   0   |   0.999326109886  |
    |   2    |   8   | 0.000261707202299 |
    |   3    |   0   |   0.997855246067  |
    |   3    |   8   |  0.00118813838344 |
    |   4    |   0   |   0.997171103954  |
    |   4    |   6   |  0.00115600414574 |
    |  ...   |  ...  |        ...        |
    +--------+-------+-------------------+
    [20000 rows x 3 columns]

    Evaluate the classifier on the test data. Default metrics are `accuracy` and
    `confusion_matrix`.

    >>> eval_ = m.evaluate(test_data)
    >>> eval_
    {'accuracy': 0.979200005531311, 'confusion_matrix':
     +--------------+-----------------+-------+
     | target_label | predicted_label | count |
     +--------------+-----------------+-------+
     |      0       |        0        |  969  |
     |      2       |        0        |   2   |
     |      5       |        0        |   2   |
     |      6       |        0        |   9   |
     |      7       |        0        |   1   |
     |      9       |        0        |   2   |
     |      1       |        1        |  1126 |
     |      2       |        1        |   2   |
     |      6       |        1        |   2   |
     |      7       |        1        |   3   |
     |     ...      |       ...       |  ...  |
     +--------------+-----------------+-------+
     [64 rows x 3 columns]}
    """

    _mt._get_metric_tracker().track('toolkit.classifier.neuralnet_classifier.create')

    # check user input: "features"
    if features is not None:
        if not isinstance(features, list):
            raise TypeError('argument features must be a list')
        else:
            for f in features:
                if not f in dataset.column_names():
                    raise RuntimeError('Feature column %s not found in dataset' % f)

    # use the default network
    if network is None:
        _network = _deeplearning.create(dataset, target, features)
    else:
    # use user specified network
        _network = _copy(network)
        input_shape = _deeplearning._get_input_shape(dataset, target, features)
        output_shape = _deeplearning._get_output_shape(dataset[target])
        # verify the netwo
        _network.verify()
        try:
            _network.verify(input_shape, output_shape)
        except ValueError as e:
            msg = ("The input network is valid, but is not compatible with "
                   "the input and output shape of the dataset.\n"
                   "Please change the network and use net.verify() with "
                   "input_shape=%s, and output_shape=%s" % (str(input_shape), str(output_shape)))
            raise ValueError(e.message + "\n" + msg)

    # update network params
    for k, v in _six.iteritems(kwargs.copy()):
        if type(v) in [int, float, bool, str, _Image]:
            if k in _network.params:
                _logging.getLogger(__name__).warning("Network parameter %s is overwritten by argument in create()" % k)
            if type(v) is bool:
                v = int(v)
            _network.params[k] = v
        elif k == 'metric' and type(v) is list:
            v = ','.join(v)
            _network.params[k] = v

    if (verbose):
        print("Using network:\n%s" % _network.__str__())


    model_name_str = 'neuralnet_classifier_v2'


    model = _sl.create(dataset=dataset,
                       target=target,
                       features=features,
                       model_name=model_name_str,
                       validation_set=validation_set,
                       verbose=verbose,
                       neuralnet_conf=_network.__config_str__(),
                       _model_constructor=_cloudpickle.dumps(NeuralNetClassifier(None)._get_wrapper()),
                       max_iterations=max_iterations,
                       class_weights=class_weights,
                       **(_network.params))

    return NeuralNetClassifier(model.__proxy__)
