__all__ = ['NeuralNet', 'MultiLayerPerceptrons', 'ConvolutionNet']

from . import layers as layers
from .layers import _LayerBase, _layer_from_string, types
from .. _main import ToolkitError as _ToolkitError
from graphlab.util import  _make_internal_url
from graphlab.connect import main as _gl_connect
from graphlab.data_structures.image import Image as _Image
import graphlab.generic as _generic
import graphlab.connect as _mt

import six as _six

class _LayerList(list):
    """ Like regular list, but has a different str and repr function """
    def __new__(cls, data=None):
        obj = super(_LayerList, cls).__new__(cls, data)
        return obj

    def __repr__(self):
        if len(self) == 0:
            return '[]'
        else:
            layer_index_added = [('layer[%d]: ' % i) + self[i].__repr__() for i in range(len(self))]
            return '\n'.join(layer_index_added)

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        return _LayerList(list(self) + list(other))


class NeuralNet(_generic.CommonEqualityMixin):
    """
    NeuralNet defines the architecture and learning parameters
    of a neural network.

    It has two properties: `layers`, and `params`. The architecture
    is represented as a list of "Layer" objects in the "layers" property.
    The global parameters are specified in `params` as a dictionary.

    This constructor should only be called directly if you are building a custom
    network from scratch. Instead, please use :py:func:`~graphlab.deeplearning.load` or
    :py:func:`~graphlab.deeplearning.loads`.

    Examples
    --------
    >>> import graphlab as gl
    >>> dataset = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')
    >>> net = gl.deeplearning.create(dataset, 'label')
    >>> net.layers
    layer[0]: ConvolutionLayer
      stride = 2
      num_channels = 10
      kernel_size = 3
    layer[1]: MaxPoolingLayer
      stride = 2
      kernel_size = 3
    layer[2]: FlattenLayer
    layer[3]: FullConnectionLayer
      num_hidden_units = 100
    layer[4]: RectifiedLinearLayer
    layer[5]: DropoutLayer
      threshold = 0.5
    layer[6]: FullConnectionLayer
      num_hidden_units = 10
    layer[7]: SoftmaxLayer
    >>> net.params
    {'learning_rate': 0.001, 'momentum': 0.9}

    See Also
    --------
    layers, params
    graphlab.deeplearning.create
    graphlab.neuralnet_classifier.create
    """
    def __init__(self, url=None, conf_str=None):
        """
        This constructor should not be called directly.
        Instead, please use :py:func:`~graphlab.deeplearning.load` or
        :py:func:`~graphlab.deeplearning.loads`.

        Constructs a NeuralNet from URL or configuration string.
        If neither parameters are provided, by default creates an empty
        NeuralNet.

        Parameters
        ----------
        url : str, optional
          The URL to a configuration file of the NeuralNet.
        conf_str : str, optional
          The configuration string of the NeuralNet.
        """
        self._layers = _LayerList()
        self._learning_params = {'learning_rate': 0.001, 'momentum': 0.9}
        if url is not None:
            self._load(_make_internal_url(url))
        if conf_str is not None:
            self._loads(conf_str)
            self.verify()

    def verify(self, input_shape=None, output_shape=None):
        """
        Verify that the layers and parameters form a valid
        neural netowrk.

        The function check a set of predefined rules, and throws
        exception if any of the rules is violated.

        Parameters
        ----------
        input_shape : (int,int,int), optional
            The shape of the input data as a three dimensional
            tuple. For image data, the tuple represents (width,height
            and channels). For flat numerical data, the tuple
            shall be (1,1,num_features).
            If ommited, assume the first layer of the network
            is compatible with the input.

        output_shape : int, optional
            The number of output classes. If ommited, assume
            the last layer of the network is compatible with the output.

        Returns
        -------
        out : bool
            True if the network passes validation

        Raises
        ------
        ValueError
            If any layer violates the predefined rules of a valid network.

        Examples
        --------
        >>> import graphlab as gl
        >>> net = gl.deeplearning.get_builtin_neuralnet('mnist')
        >>> net.verify()
        True
        >>> del net.layers[2]
        >>> net.verify()
        ValueError: Layer 3: Must flatten layer before giving to
        fully-connected layer.
        """
        _layers = self._layers
        outputs = [None] * (len(_layers) + 1)

        # Check if network input channels match data input size channels
        if 'input_shape' in self.params and input_shape is not None:
            _network_input_channels = int(self.params['input_shape'].split(',')[0])
            if _network_input_channels != input_shape[2]:
                raise ValueError("Number of input channels in parameter 'input_size' " +
                        "does not match numer of channels in input data")
        # Initialize first layer input size,

        outputs[0] = input_shape

        # Check if network has size 0

        if len(_layers) == 0:
            raise ValueError("Cannot have a network of size 0")

        # Check if input is flat or 2-d. If there is no input, assume input matches flatness of first layer.
        if input_shape is None:
            if _layers[0]._type not in (types.CONVOLUTION_LAYER, types.POOLING_LAYER,
                                        types.TRANSITION_LAYER):
                conv2flat = True
            else:
                conv2flat = False
        elif input_shape[0] == 1 and input_shape[1] == 1:
            conv2flat = True
        else:
            conv2flat = False
            if (_layers[0]._type not in (types.CONVOLUTION_LAYER, types.POOLING_LAYER, types.TRANSITION_LAYER)) or _layers[0]._name == "lrn":
                raise ValueError('Input data needs flattening. Try adding a flatten layer at the beginning of the network.')


        # Check if first layer is activation layer.
        if _layers[0]._type == types.ACTIVATION_LAYER:
            raise ValueError(("First layer cannot be an activation layer.\n"
                             "Try FullConnectionLayer, ConvolutionLayer or FlattenLayer"))

        # First layer cannot be output layer.
        if _layers[0]._type == types.OUTPUT_LAYER:
            raise ValueError(("First layer cannot be an output layer.\n"
                             "Try FullConnectionLayer, ConvolutionLayer or FlattenLayer"))



        # For classification last layer must be output layer, and must be preceeded by fully connected layer. For regression we only support one output unit.
        if output_shape is not None and output_shape >= 2:
            if _layers[-1]._type != types.OUTPUT_LAYER:
                raise ValueError("Last layer must be an output layer")
            elif _layers[-2]._type != types.CONNECTION_LAYER:
                raise ValueError("Second last layer must be a FullConnectionLayer")

        # Last fully connected layer must have number of units that matches number of output classes
        if output_shape is not None and output_shape >= 2 and _layers[len(_layers) - 2].num_hidden_units != output_shape:
            raise ValueError(("The last FullConnectionLayer must have the same number of hidden units\n"
                             " as the number of output classes. Try setting num_hidden_units to %d." % output_shape))

        # Scan through layers
        for i in range(0, len(_layers)):
            # verify the layer parameter
            try:
                _layers[i]._verify()
            except ValueError as e:
                raise ValueError('Layer %i: %s' % (i, str(e)))

            # Check for multiple activations in a row, we already checked first layer is not activation
            if _layers[i]._type == types.ACTIVATION_LAYER and _layers[i - 1]._type == types.ACTIVATION_LAYER:
                raise ValueError("Layer %i: Cannot have two activation layers in a row." % i)

            # Check if there is an output layer not at the end of the network
            if _layers[i]._type == types.OUTPUT_LAYER and i != len(_layers) - 1:
                raise ValueError("Layer %i: Output layer cannot be followed by any other layer." % i)

            # Check if flatten layer is flattening data that is already flat
            if _layers[i]._type == types.TRANSITION_LAYER:
                if conv2flat:
                    raise ValueError("Layer %i: Data is already flat. Try removing the FlattenLayer." % i)
                else:
                    conv2flat = True

            # Check if network is trying to convolve flat data.
            if _layers[i]._type in (types.CONVOLUTION_LAYER, types.POOLING_LAYER) and conv2flat:
                raise ValueError("Layer %i: Cannot convolve or pool flattened inputs." % i)

            # Check if network is trying to perfrom local response normalization on flattened input

            if _layers[i]._name == "lrn" and conv2flat:
                raise ValueError("Layer %i: Connot perform local response normalization on flattened input." %i)

            # Check if fully connected layer is not taking flat input
            if _layers[i]._type == types.CONNECTION_LAYER and not conv2flat:
                raise ValueError(("Layer %i: Data needs flattening.\n" "Try adding a FlattenLayer between "
                                  "ConvolutionLayer/PoolingLayer and FullConnectionLayer." % i))

            # Check if kernel size is compatible with input size for pooling and convolution
            if _layers[i]._type in (types.CONVOLUTION_LAYER, types.POOLING_LAYER):
                if input_shape is not None:
                    if outputs[i][0] < _layers[i].kernel_size or outputs[i][1] < _layers[i].kernel_size:
                        raise ValueError(("Layer %i: Kernel size cannot exceed the size of the input of previous layer.\n"
                                         "Try either increasing padding in the previous layer, "
                                         "or decreasing the kernel_size." % i))

            # Set the input shape for the next layer
            if input_shape is not None:
                outputs[i + 1] = _layers[i]._get_output_shape(outputs[i])

            # Check if num_groups is divisible by input size or output_size
            if (_layers[i]._type == types.CONVOLUTION_LAYER):
                if outputs[i] is not None:
                    if hasattr(_layers[i], 'num_groups'):
                        ngroup = _layers[i].num_groups
                    else:
                        ngroup = layers._LayerDefaults().NUMGROUPS_DEFAULT
                    if outputs[i][2] % ngroup != 0:
                        raise ValueError('Layer %i: Input channels must divide group size' % i)
                    if outputs[i + 1][2] % ngroup != 0:
                        raise ValueError('Layer %i: Output channels must divide group size' % i)

            # Check if Local Response Normalization layer has a local size smaller than number of input channels
            if(_layers[i]._name == 'lrn'):
                if outputs[i] is not None:
                    if hasattr(_layers[i], 'local_size'):
                        l_size = _layers[i].local_size
                    else:
                        l_size = layers._LayerDefaults().LOCALSIZE_DEFAULT
                    if outputs[i][2] < l_size:
                        raise ValueError("Layer %i: 'local_size' needs to be smaller than num_channels in the previous layer" % i)
        return True

    def save(self, url):
        """
        Save the neuralnet to url.

        Parameters
        ----------
        url : str
            The URL to save the network.

        Examples
        --------
        >>> import graphlab as gl
        >>> net = gl.deeplearning.get_builtin_neuralnet('mnist')
        >>> net.save('mnist.conf')

        See Also
        --------
        graphlab.deeplearning.load
        """
        _gl_connect.get_unity().__write__(_make_internal_url(url), self.__config_str__())

    @property
    def layers(self):
        """
        A list of Layer objects defining the architecture.

        You can modify the layers using list operations like append, del, insert
        and assign.

        Examples
        --------
        >>> import graphlab as gl
        >>> net = gl.deeplearning.get_builtin_neuralnet('mnist')

        Inspect the layers

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

        Modify the layers

        >>> # remove a new layer
        >>> del net.layers[-1]
        >>> # add a new layer
        >>> net.layers.append(gl.deeplearning.layers.SoftmaxLayer())
        >>> # modify properties of a layer
        >>> net.layers[4].num_hidden_units=50

        See Also
        --------
        verify
        graphlab.deeplearning.layers
        """
        return self._layers

    @layers.setter
    def layers(self, value):
        if not isinstance(value, list):
            raise TypeError('layers must be list type')
        self._layers = _LayerList(value)

    @property
    def params(self):
        """
        A dictionary of hyper parameters used for training a classifier.

        By default, the dictionary only contains most commonly
        used hyperparameters: `learning_rate`, and `momentum`.
        Unspecified hyperparamters will have their default values.

        The possible hyper parameters can be queried from
        :func:`graphlab.neuralnet_classifier.get_default_options`.

        To override the default hyperparamters, simply set its value
        in this property. You can also override the hyperparameters
        when calling :func:`graphlab.neuralnet_classifier.create`.

        Examples
        --------
        >>> import graphlab as gl
        >>> gl.neuralnet_classifier.get_default_options()
        +---------------------------+---------------+--------------------------------+
        |            name           | default_value |          description           |
        +---------------------------+---------------+--------------------------------+
        |        init_random        |    gaussian   | Initialization method of t ... |
        |     learning_rate_step    |       1       |       learning rate step       |
        |   model_checkpoint_path   |               | The path for saving model  ... |
        | learning_rate_start_epoch |       0       | start learning rate schedu ... |
        |       learning_rate       |     0.001     | global learning_rate for b ... |
        |       subtract_mean       |       1       | subtract mean image from t ... |
        |         batch_size        |      100      |           batch size           |
        |     min_learning_rate     |     1e-05     |     minimun learning rate      |
        | model_checkpoint_interval |       5       |  Save model every K iteration  |
        |    learning_rate_gamma    |      0.1      | learning rate decay parame ... |
        |       random_mirror       |       0       |  randomly mirroring the image  |
        |   bias_l2_regularization  |       0       |    regularization for bias     |
        |    learning_rate_alpha    |      0.5      | learning rate decay parame ... |
        |           device          |      auto     | hardware device for runnin ... |
        |        random_crop        |       0       |  randomly cropping the image   |
        |   learning_rate_schedule  |    constant   | learning rate scheduling a ... |
        |     l2_regularization     |     0.0005    | global regularization for  ... |
        |           metric          |      auto     |    metric(s) for evaluation    |
        |          momentum         |      0.9      | global momentum for bias a ... |
        |       max_iterations      |       10      |      number of iterations      |
        +---------------------------+---------------+--------------------------------+

        >>> net = gl.deeplearning.get_builtin_neuralnet('mnist')

        Inspect the hyperparameterss

        >>> net.params
        {'learning_rate': 0.001, 'momentum': 0.9}

        Override hyperparameters for this network

        >>> net.params['learning_rate'] = 0.1
        >>> net.params['init_random'] = 'xavier'

        See Also
        --------
        graphlab.neuralnet_classifier.create
        graphlab.neuralnet_classifier.get_default_options
        """
        return self._learning_params

    @params.setter
    def params(self, value):
        self._learning_params = value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        ss = "\n### network layers ###\n"
        ss += self.layers.__repr__()
        ss += "\n### end network layers ###\n"
        ss += "\n### network parameters ###\n"
        for k, v in _six.iteritems(self.params):
            #Image type parameter is not well captured by string representation.
            if type(v) is not _Image:
                ss += str(k) + ' = ' + str(v) + '\n'
        ss += "### end network parameters ###\n"
        return ss

    def __copy__(self):
        return NeuralNet(conf_str=self.__config_str__())

    def __config_str__(self):
        begin = "netconfig = start\n"
        body = ""
        for i in range(0, len(self._layers)):
            layer_str = "layer[%d->%d] = %s" % (i, i + 1, self._layers[i].__str__())
            body += layer_str
        end = "netconfig = end\n"
        netstr = begin + body + end

        paramstr = "\n## network parameters\n"
        for k, v in _six.iteritems(self.params):
            # Image type parameter is not well captured in string representation.
            if type(v) is not _Image:
                paramstr += str(k) + ' = ' + str(v) + '\n'
        paramstr += "## end network parameters\n"
        return netstr + paramstr

    def _load(self, url):
        """ Internal. Load the network from url. """
        conf_str = _gl_connect.get_unity().__read__(url)
        self._loads(conf_str)

    def _loads(self, conf_str):
        self._layers = []
        self._learning_params = {}
        self._parse(conf_str)

    def _parse(self, conf_str):
        """ Parse the config string and update the layers and parameters"""
        parser = _NeuralNetParser()
        try:
            self.layers, self.params = parser.parse(conf_str)
        except Exception as e:
            raise ValueError('Malformed config file. ' + str(e))


class MultiLayerPerceptrons(NeuralNet):
    """
    A specific NeuralNet architecture for general classification problem.

    In this architecture, the layers alternates between
    :py:func:`FullConnectionLayer` and :py:func:`SigmoidLayer` (or other
    activation layers).

    Parameters
    ----------
    num_hidden_layers : int
        The number of hidden layers in the network.
    num_hidden_units : list[int]
        A list with size equal to num_hidden_layers, and specify
        the number of hidden units for each hidden layer
    activation : {'sigmoid', 'tanh', 'relu', 'softplus'}, optional
        Choice of the activation function.
    input_dimension : int, either 1 or 2, optional.
        The dimension of input data. For numerical data, the
        input dimension is 1 (by default); for image data, the
        input dimension is 2, and the first layer of the returned
        network will be :class:`~graphlab.deeplearning.layers.FlattenLayer`.

    Examples
    --------

    >>> net = graphlab.deeplearning.MultiLayerPerceptrons(num_hidden_layers=2,
                                                          num_hidden_units=[5,5])
    >>> net
    ### network layers ###
    layer[0]: FullConnectionLayer
      num_hidden_units = 5
    layer[1]: SigmoidLayer
    layer[2]: FullConnectionLayer
      num_hidden_units = 5
      layer[3]: SoftmaxLayer
    ### end network layers ###
    <BLANKLINE>
    ### network parameters ###
    learning_rate = 0.001
    momentum = 0.9
    ### end network parameters ###

    See Also
    --------
    graphlab.deeplearning.layers

    Notes
    -----
    The last hidden layer is used for output.
    If your input is image, set ``input_dimension`` to 2.

    References
    ----------
    - `Tutorial on Multilayer Perceptron
      <http://www.deeplearning.net/tutorial/mlp.html#mlp>`_
    """
    def __init__(self, num_hidden_layers, num_hidden_units, input_dimension=1,
                 activation='sigmoid', **kwargs):
        super(self.__class__, self).__init__()
        _mt._get_metric_tracker().track('toolkit.deeplearning.MultiLayerPerceptrons')
        if input_dimension not in (1, 2):
            raise ValueError('input_dimension must be either 1 or 2')

        ActivationLayer = _activation_layer_from_string(activation)
        if num_hidden_layers != len(num_hidden_units):
            raise ValueError('Length of num_hidden_units must equal to num_hidden_layers')
        for i in range(0, num_hidden_layers - 1):
            self._layers.append(layers.FullConnectionLayer(num_hidden_units[i]))
            self._layers[-1]._set_params(**kwargs)
            self._layers.append(ActivationLayer())
            self._layers[-1]._set_params(**kwargs)

        # output layer
        if (num_hidden_units[-1] > 0):
            self._layers.append(layers.FullConnectionLayer(num_hidden_units[-1]))
            self._layers[-1]._set_params(**kwargs)
            if (num_hidden_units[-1] > 1):
                self._layers.append(layers.SoftmaxLayer())

        # input layer
        if (input_dimension == 2):
            self._layers.insert(0, layers.FlattenLayer())


class ConvolutionNet(NeuralNet):
    """
    A specific NeuralNet architecture used in image classification problem.

    In this architecture, the layers alternates among
    :py:class:`~graphlab.deeplearning.ConvolutionLayer`,
    and :py:class:`~graphlab.deeplearning.MaxPoolingLayer` (or other
    pooling layers).

    If num_output_units is greater than zero, a
    :py:class:`~graphlab.deeplearning.FlattenLayer`,
    :py:class:`~graphlab.deeplearning.FullConnectionLayer`,
    with `num_output_units` hidden units,
    and
    :py:class:`~graphlab.deeplearning.SoftmaxLayer`
    will be added as the final output layer.

    If num_output_units is zero, the last layer is a
    :py:class:`~graphlab.deeplearning.FlattenLayer`, which
    user defined output layer can be connected to.

    Parameters
    ----------
    num_convolution_layers: int
        Number of convolution layers
    kernel_size: int
        Size of the convolution kernel for the convolution and pooling layer.
    num_channels: int
        Number of output channels of each convolution layer (also called features maps).
    num_output_units : int
        Number of units in the final output layer
    pooling: {'max_pooling', 'sum_pooling', 'avg_pooling'}, optional
        Choice of the pooling method.

    Examples
    --------
    >>> net = graphlab.deeplearning.ConvolutionNet(num_convolution_layers=2,
                                                   kernel_size=3, stride=2,
                                                   num_channels=10,
                                                   num_output_units=10)
    >>> net
    ### network layers ###
    layer[0]: ConvolutionLayer
      stride = 2
      num_channels = 10
      kernel_size = 3
    layer[1]: RectifiedLinearLayer
    layer[2]: MaxPoolingLayer
      stride = 2
      kernel_size = 3
    layer[3]: ConvolutionLayer
      stride = 2
      num_channels = 10
      kernel_size = 3
    layer[4]: RectifiedLinearLayer
    layer[5]: MaxPoolingLayer
      stride = 2
      kernel_size = 3
    layer[6]: FlattenLayer
    layer[7]: FullConnectionLayer
      num_hidden_units = 100
    layer[8]: SoftmaxLayer
    ### end network layers ###
    <BLANKLINE>
    ### network parameters ###
    learning_rate = 0.001
    momentum = 0.9
    ### end network parameters ###

    References
    ----------
    - `Tutorial on Convolutional Neural Networks
      <http://www.deeplearning.net/tutorial/lenet.html>`_

    - Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
      classification with deep convolutional neural networks." Advances in
      neural information processing systems. 2012.

    - LeCun, Yann, and Yoshua Bengio. "Convolutional networks for images,
      speech, and time series." The handbook of brain theory and neural
      networks 3361 (1995).
    """
    def __init__(self, num_convolution_layers,
                 kernel_size,
                 num_channels,
                 num_output_units,
                 stride=layers._LayerDefaults.STRIDE_DEFAULT,
                 pooling='max_pooling',
                 **kwargs):
        _mt._get_metric_tracker().track('toolkit.deeplearning.ConvolutionNet')
        super(self.__class__, self).__init__()
        PoolingLayer = _pooling_layer_from_string(pooling)
        for n in range(num_convolution_layers):
            self._layers.append(layers.ConvolutionLayer(kernel_size=kernel_size,
                                                        num_channels=num_channels,
                                                        stride=stride))
            self._layers[-1]._set_params(**kwargs)
            self._layers.append(PoolingLayer(kernel_size=kernel_size,
                                             stride=stride))
            self._layers[-1]._set_params(**kwargs)
        self._layers.append(layers.FlattenLayer())
        self._layers[-1]._set_params(**kwargs)

        # output layer
        if (num_output_units > 0):
            self._layers.append(layers.FullConnectionLayer(num_output_units))
            self._layers[-1]._set_params(**kwargs)
            if (num_output_units > 1):
                self._layers.append(layers.SoftmaxLayer())


class _NeuralNetParser(object):
    """
    Parser for the neuralnet configuration file.
    """
    def __init__(self):
        self.scopes = ['global']
        self.layers = []
        self.params = {}
        self.current_scope = self.params

    def update_current_scope(self, k, v):
        if isinstance(self.current_scope, _LayerBase):
            setattr(self.current_scope, k, v)
        elif isinstance(self.current_scope, dict):
            self.current_scope[k] = v

    def _parse_key_value(self, line):
        # parse k = v
        kv = line.split('=')
        assert len(kv) == 2
        k = kv[0].strip()
        v = kv[1].strip()
        try:
            v = int(v)
        except:
            try:
                v = float(v)
            except:
                pass
        return k, v

    def _parse_layer(self, line):
        # layer[from->to]
        from_to = line[(line.find('[') + 1):line.find(']')].split('->')
        _from = int(from_to[0])
        _to = int(from_to[1])
        return _from, _to

    def _parse_line(self, line):
        if line.startswith('#') or len(line) == 0:
            return
        else:
            k, v = self._parse_key_value(line)
            if k == 'netconfig' and v == 'start':
                self.scopes.append('netconfig')
                self.current_scope = None
            elif k == 'netconfig' and v == 'end':
                assert self.scopes[-1] == 'layer'
                self.scopes.pop()
                assert self.scopes[-1] == 'netconfig'
                self.scopes.pop()
                assert len(self.scopes) == 1
                assert self.scopes[-1] == 'global'
                self.current_scope = self.params
            elif k.startswith('layer'):
                _from, _to = self._parse_layer(k)
                layer = _layer_from_string(v)
                self.layers.insert(_from, layer)
                if self.scopes[-1] == 'layer':
                    self.scopes.pop()
                assert self.scopes[-1] == 'netconfig'
                self.scopes.append('layer')
                self.current_scope = layer
            else:
                self.update_current_scope(k, v)

    def parse(self, conf_str):
        lines = conf_str.split('\n')
        for l in lines:
            try:
                self._parse_line(l.strip())
            except Exception:
                raise RuntimeError('Error at: ' + l)
        assert len(self.scopes) == 1 and self.scopes[-1] == 'global', 'Scope mismatch'
        assert len(self.layers) > 0, 'No layer in the config'
        return self.layers, self.params


def _activation_layer_from_string(activation):
    if activation == 'sigmoid':
        return layers.SigmoidLayer
    elif activation == 'tanh':
        return layers.TanhLayer
    elif activation == 'relu':
        return layers.RectifiedLinearLayer
    elif activation == 'softplus':
        return layers.SoftplusLayer
    else:
        supported_activations = ['sigmoid', 'tanh', 'relu', 'softplus']
        raise _ToolkitError('Unknown activation layer %s. Supported activation layers are: %s' % (activation, str(supported_activations)))


def _pooling_layer_from_string(pooling):
    if pooling == 'max_pooling':
        return layers.MaxPoolingLayer
    elif pooling == 'sum_pooling':
        return layers.SumPoolingLayer
    elif pooling == 'avg_pooling':
        return layers.AveragePoolingLayer
    else:
        supported_poolings = ['max_pooing', 'sum_pooling', 'avg_pooling']
        raise _ToolkitError('Unknown pooling layer %s. Supported pooling layers are: %s' % (pooling, str(supported_poolings)))
