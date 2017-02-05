import array
from graphlab.data_structures.image import Image
from . _neuralnets import MultiLayerPerceptrons, ConvolutionNet, NeuralNet
from . import layers
from . import _builtin_neuralnets
import graphlab.connect as _mt
import copy

def _get_input_shape(dataset, target, features):
    """
    Return the input layer shape of the SFrame as a 3 dimensional tuple.
    For dense numerical columns, the shape is (1,1,num_features).
    For image column, the shape is (width,height,channels).
    """
    if features is not None:
        column_names = dataset[features].column_names()
        column_types = dataset[features].column_types()
    else:
        column_names = dataset.column_names()
        column_types = dataset.column_types()
    ret = [1, 1, 0]

    dense_vec_input = False
    image_input = False

    for (_name, _type) in zip(column_names, column_types):
        if dense_vec_input and image_input:
            raise ValueError("With graphlab.Image type input, can only have one feature column.\
                    Try removing either Image column or other features columns")
        if _name == target:
            continue
        elif _type in (int, float):
            ret[2] += 1
            dense_vec_input = True
        elif _type == array.array:
            ret[2] += len(dataset[_name][0])
            dense_vec_input = True
        elif _type == Image:
            img = dataset[_name][0]
            ret[0] = img.width
            ret[1] = img.height
            ret[2] = img.channels
            image_input = True
    return ret


def _get_output_shape(labels, objective='classification'):
    """
    Return the output layer shape of the label SArray.

    For classification problem, the label SArray is assumed be int type,
    and has value range (0, K-1) where K is the number of classes in the columns.

    Otherwise, we treat int or float SArray as regression target, and
    return output_shape = 1.
    """
   # if (labels.dtype() == int):
    num_unique = len(labels.unique())
   #     if (labels.min() == 0 and labels.max() == num_unique - 1):
    return num_unique


def _create_conv_net(input_shape, output_shape, data_size, **kwargs):
    """
    Return a simple 1 layer convolution net with matching output shape.
    """
    default_kernel_size = 3
    default_stride = 2
    default_channels = 10
    default_hidden_units = 100
    ret = ConvolutionNet(num_convolution_layers=1,
                         kernel_size=default_kernel_size,
                         stride=default_stride,
                         num_channels=default_channels, num_output_units=0)
    ret.layers.append(layers.FullConnectionLayer(default_hidden_units))
    ret.layers.append(layers.RectifiedLinearLayer())
    ret.layers.append(layers.DropoutLayer())
    ret.layers.append(layers.FullConnectionLayer(output_shape))
    if (output_shape >= 2):
        ret.layers.append(layers.SoftmaxLayer())
    return ret


def _create_perceptron_net(input_shape, output_shape, data_size, input_dimension, **kwargs):
    """
    Return a simple 1 layer perceptron net with matching output shape
    """
    assert input_dimension in (1, 2)

    if input_dimension == 1:
        assert input_shape[0] == 1 and input_shape[1] == 1

    ret = MultiLayerPerceptrons(num_hidden_layers=2,
                                num_hidden_units=[10, output_shape],
                                input_dimension=input_dimension)
    return ret


def create(dataset, target, features=None, network_type='auto'):
    """
    Creates a :class:`~graphlab.deeplearning.NeuralNet` given the input dataset.

    The returned :class:`~graphlab.deeplearning.NeuralNet`
    can be provided to :func:`graphlab.neuralnet_classifier.create`
    to train a :class:`~graphlab.neuralnet_classifier.NeuralNetClassifier`.

    The general workflow involves three steps:

    1. Create a neuralnet using :func:`graphlab.deeplearning.create`
    2. Tweak the neuralnet parameters so that they are appropriate for the
       desired task
    3. Pass the net to :func:`graphlab.neuralnet_classifier.create` to train
       a :class:`~graphlab.neuralnet_classifier.NeuralNetClassifier`.
       with the given network architechture and hyperparameters.

    When data contains image column, returns a :class:`~graphlab.deeplearning.ConvolutionNet`,
    otherwise returns a :class:`~graphlab.deeplearning.MultiLayerPerceptrons`.


    Parameters
    ----------
    dataset : SFrame
        The dataset for to create a NeuralNet for.
    target : str
        The name of the target column to be predicted
    features : list[str], optional
        Column names of the features used to train the model. Each column must
        contain vectors of floats or there can be one column of Image type.
        The default argument is ``None``, which means all columns are used,
        except the target.
    network_type : str, {'auto, 'perceptrons', 'convolution'}, optional
        Specify the type of network to create. Default 'auto' will create
        a :class:`~graphlab.deeplearning.ConvolutionNet` for image input,
        and a :class:`~graphlab.deeplearning.MultiLayerPerceptrons` for regular
        numerical input.

    Returns
    -------
    out : NeuralNet

    Examples
    --------
    >>> import graphlab as gl
    >>> dataset = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')
    >>> net = gl.deeplearning.create(dataset, 'label')
    >>> model = gl.neuralnet_classifier.create(dataset, 'label', network=net)

    Notes
    -----
    The network returned by this function might be suboptimal. It is expected
    to be used as an initial starting point for training
    :class:`graphlab.neuralnet_classifier.NeuralNetClassifier`.
    """
    _mt._get_metric_tracker().track('toolkit.deeplearning.create')
    input_shape = _get_input_shape(dataset, target, features)
    output_shape = _get_output_shape(dataset[target])
    if network_type not in {'auto', 'perceptrons', 'convolution'}:
        raise ValueError('network_type %s is not supported.' % network_type)
    nnet = None
    if (input_shape[0] == 1 and input_shape[1] == 1):
        if (network_type == 'auto' or network_type == 'perceptrons'):
            nnet = _create_perceptron_net(input_shape, output_shape, len(dataset),
                                          input_dimension=1)
        else:
            raise ValueError('Cannot convolution net for numerical data. Expect image data.')
    else:
        if network_type == 'auto' or network_type == 'convolution':
            nnet = _create_conv_net(input_shape, output_shape, len(dataset))
        else:
            nnet = _create_perceptron_net(input_shape, output_shape, len(dataset),
                                          input_dimension=2)
    nnet.verify(input_shape, output_shape)
    return nnet


def load(url):
    """
    Load a saved NeuralNet object from url.

    Parameters
    ----------
    url : str
        The url that points to a saved NeuralNet object.

    See Also
    --------
    graphlab.deeplearning.NeuralNet.save

    Returns
    -------
    out : NeuralNet

    Examples
    --------
    >>> import graphlab as gl
    >>> net = gl.deeplearning.get_builtin_neuralnet('mnist')
    >>> net.save('mnist.conf')
    >>> net2 = gl.deeplearning.load('mnist.conf')
    """
    return NeuralNet(url)


def loads(conf_string):
    """
    Load a saved NeuralNet object from string.

    Parameters
    ----------
    conf_string : str
        The serialization of a NeuralNet object.

    Returns
    -------
    out: NeuralNet

    See Also
    --------
    graphlab.deeplearning.NeuralNet.save

    Examples
    --------
    >>> import graphlab as gl
    >>> net = gl.deeplearning.get_builtin_neuralnet('mnist')
    >>> net.save('mynet')
    >>> config_string = file('mynet').read()
    >>> net2 = gl.deeplearning.loads(config_string)
    """
    return NeuralNet(conf_str=conf_string)

def get_builtin_neuralnet(arch):
    """
    Returns neural network architechtures for the MNIST and Imagenet datasets.
    See References for detailed explanations of the architechtures.

    Parameters
    ----------
    arch: {'imagenet' | 'mnist'}
        Specifies the network architechture to be returned

    Returns
    -------
    out: NeuralNet

    References
    ----------
    - LeCun, Yann, et al. "Gradient-based learning applied to document
      recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

    - Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
      classification with deep convolutional neural networks." Advances in
      neural information processing systems. 2012.

    Examples
    --------

    >>> net = graphlab.deeplearning.get_builtin_neuralnet('mnist')

    """
    _mt._get_metric_tracker().track('toolkit.deeplearning.get_builtin_neuralnet')

    if arch == 'mnist':
        return copy.copy(_builtin_neuralnets._MNIST_NET)
    elif arch == 'imagenet':
        return copy.copy(_builtin_neuralnets._IMAGENET_NET)
    else:
        raise ValueError("Not a valid net. Please pass in either 'mnist' or 'imagenet'")

