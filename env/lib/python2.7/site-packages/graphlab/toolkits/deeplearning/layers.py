"""
The layers module contains classes for each of the layer types supported by the
Deeplearning toolkit. This module also makes it easy to add, remove, and modify
layers in a network to optimize its architecture for a particular learning
objective. The documentation for each layer lists possible and required parameters
for each layer type, as well as rules for network validity.

**Building a custom network**

A network's architecure is defined by a list of layer objects.

>>> custom_net_layers = list()
>>> custom_net_layers.append(graphlab.deeplearning.layers.ConvolutionLayer(kernel_size=3,
                                                                    stride=2,
                                                                    num_channels=10))
>>> custom_net_layers.append(graphlab.deeplearning.layers.FullConnectionLayer(10))
>>> custom_net_layers.append(graphlab.deeplearning.layers.SoftmaxLayer())

Now we create a :class:`~graphlab.deeplearning.NeuralNet`
object to encapsulate this list.

>>> custom_net = graphlab.deeplearning.NeuralNet()
>>> custom_net.layers = custom_net_layers

Now, we want to verify that our constructed architecture is correct.

>>> custom_net.verify()

This breaks, and gives an error message that we cannot connect a fully connected
layer to a non-flat layer, in this case the Convolution Layer. So we modify
the network, something made easy by the Deep Learning toolkit in GraphLab
Create.

>>> custom_net.layers.insert(1, graphlab.deeplearning.layers.FlattenLayer())
>>> custom_net.verify()

**Layer types**

All of the layer classes in Graphlab Create can conceptually be organized into
several functional groups: connection layers, convolution layers, activation layers,
pooling layers, output layers, regularization layers, and transition layers.

Connection layers generate edges between the previous layer and the next. These
edges correspond to parameter values learned by the network during the training
procedure. Currently, the connection layer offered is
:class:`~graphlab.deeplearning.FullConnectionLayer`

Convolution layers are similar in purpose to connection layers, except they
generate the connection via convolution. This reduces the number of parameters
to learn, and encodes the idea of detecting local patterns in the previous
layer, where we consider local patterns to be patterns involving units that are
spatially close. Currently, the convolution layer offered is
:class:`~graphlab.deeplearning.ConvolutionLayer`

Activation layers transform each output of the previous layer with some specified
function. Therefore, the output size of the activation layer is
identical to the output size of the previous layer. This is where the
non-linearity and expressive power of neural nets come from. Currently, the
activation layers offered include :class:`~graphlab.deeplearning.SigmoidLayer`,
:class:`~graphlab.deeplearning.RectifiedLinearLayer`,
:class:`~graphlab.deeplearning.TanhLayer`,
:class:`~graphlab.deeplearning.SoftplusLayer`

Pooling layers take a 2-d input layer and reduce the by computing a reduction
function over a sliding window. This encodes both spatial and scale invariance
into the network learning, and pooling layers often follow convolution layers.
Common pooling layers include max pooling, sum pooling, and average pooling.
These are offered in GraphLab Create with the
:class:`~graphlab.deeplearning.MaxPoolingLayer`,
:class:`~graphlab.deeplearning.SumPoolingLayer`,
:class:`~graphlab.deeplearning.AveragePoolingLayer` layer classes.

Output layers are the classification layer at the end of a network. Currently,
GraphLab Create offers the :class:`~graphlab.deeplearning.SoftmaxLayer` as
an output layer.

Regularization layers are layers that aid the training procedure in various ways,
and help the neural network from overfitting to the training set. Currently,
GraphLab Create offers :class:`~graphlab.deeplearning.DropoutLayer` and
:class:`~graphlab.deeplearning.LocalResponseNormalizationLayer` layer
classes.

Transition layers assist in moving from one layer shape in the network
architechture to another. For instance, pooling and convolution layers are not
flat, while fully connection layers are. To transtion from a non-flat layer to
a flat layer, one must use a :class:`~graphlab.deeplearning.FlattenLayer`.
"""

import graphlab.generic as _generic
import six as _six

class _layer_types_enum():
    """ Layer Type Enums """
    CONNECTION_LAYER = "CONNECTION"
    ACTIVATION_LAYER = "ACTIVATION"
    CONVOLUTION_LAYER = "CONVOLUTION"
    OUTPUT_LAYER = "OUTPUT"
    POOLING_LAYER = "POOLING"
    TRANSITION_LAYER = "TRANSITION"
    REGULARIZATION_LAYER = "REGULARIZATION"


class _LayerDefaults():
    PADDING_DEFAULT = 0
    NUMGROUPS_DEFAULT = 1
    STRIDE_DEFAULT = 1
    LOCALSIZE_DEFAULT = 5

""" deeplearning.layers.types """
types = _layer_types_enum()


class _LayerBase(_generic.CommonEqualityMixin):
    """
    Base class for Layer object. Provides name, and type property and
    shared functionality for converting from and to string and cxxnet string
    representations.
    """
    _name = None
    _type = None

    def __init__(self, _name, _type):
        self._name = _name
        self._type = _type

    def __repr__(self):
        ss = [self.__class__.__name__]
        key_values = dict([(k, self.__getattribute__(k)) for k in self.__dict__ if not k.startswith('_')])
        for (k, v) in _six.iteritems(key_values):
            ss.append("  %s = %s" % (k, str(v)))
        return '\n'.join(ss)

    def __str__(self):
        ret = self._name + "\n"
        key_values = dict([(k, self.__getattribute__(k)) for k in self.__dict__ if not k.startswith('_')])
        for k, v in _six.iteritems(key_values):
            ret += "  %s = %s\n" % (k, str(v))
        return ret

    def _verify(self):
        return True

    def _get_output_shape(self, input_shape):
        return input_shape

    def _set_params(self, **kwargs):
        '''
        This is a back door to enable setting of arbitrary kwargs without
        exposing the ability to pass bogus parameters to the user.
        '''
        for (k, v) in _six.iteritems(kwargs):
            setattr(self, k, v)
        return

    @property
    def name(self):
        '''
        Return the name of the network layer.

        Returns
        -------
            out: str
                The name of the layer string

        Examples
        ----------

        >>> layer = graphlab.deeplearning.layers.FullConnectionLayer(20)
        >>> layer.name

        '''

        return self._name

    @property
    def type(self):
        '''
        Return the type of the network layer

        Returns
        -------
            out : {'CONNECTION' | 'ACTIVATION' | 'CONVOLUTION' | 'OUTPUT' | 'POOLING' | 'TRANSITION' | 'REGULARIZATION'}
                The type of the network layer

        Examples
        --------

        >>> layer = graphlab.deeplearning.layers.FullConnectionLayer(20)
        >>> layer.type

        '''
        return self._type


# /**************************************************************************/
# /*                                                                        */
# /*                           Connection Layers                            */
# /*                                                                        */
# /**************************************************************************/
class FullConnectionLayer(_LayerBase):
    """
    A connection layer, connecting every unit in the incoming layer to every
    unit defined in this layer. Each unit in the layer also has a learned bias
    added to the input values from the previous layer.

    Because the number of parameters in this layer is very high,
    it can capture very complex relationships but can be
    difficult to train. The input to this layer must be flattened.

    Parameters
    ----------
    num_hidden_units : int
        The number of hidden units in the layers.
    init_bias : int, optional
        The initial value of the bias of the layer. Default 0.
    init_sigma : float , optional. Default 0.01.
        The sigma with which to sample a Gaussian distribution if
        'init_random' is 'gaussian'.
    init_random: {'gaussian', 'xavier'}, optional. Default 'gaussian'.
        The distribution from which to initialize parameters:
        Gaussian is a normal distribution.
        Xavier determines the scale of initialization based on number
        of input and output neurons. See References for more
        information on the Xavier algorithm.

    See Also
    --------
    FlattenLayer, ConvolutionLayer

    References
    ----------
    - Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty
      of training deep feedforward neural networks." International Conference
      on Artificial Intelligence and Statistics. 2010.

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.FullConnectionLayer(20)
    """
    def __init__(self, num_hidden_units, init_bias=0, init_sigma=0.01, init_random='gaussian'):
        super(FullConnectionLayer, self).__init__("fullc", types.CONNECTION_LAYER)
        self.num_hidden_units = num_hidden_units
        self.init_bias = init_bias
        self.init_sigma = init_sigma
        self.init_random = init_random

    def _verify(self):
        if (self.num_hidden_units <= 0):
            raise ValueError('num_hidden_units must be positive')
        return super(FullConnectionLayer, self)._verify()

    def _get_output_shape(self, input_shape):
        return (self.num_hidden_units, 1, 1)


class FlattenLayer(_LayerBase):
    """
    A layer which takes a 2-d input coming from a convolution layer or
    pooling layer and flattens it into something a form a fully connected layer
    can take. It cannot flatten already flat data.

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.FlattenLayer()

    """
    def __init__(self):
        super(FlattenLayer, self).__init__("flatten", types.TRANSITION_LAYER)

    def _get_output_shape(self, input_shape):
        n = input_shape[0] * input_shape[1] * input_shape[2]
        return (n, 1, 1)


# /**************************************************************************/
# /*                                                                        */
# /*                           Activation Layers                            */
# /*                                                                        */
# /**************************************************************************/
class _ActivationLayerBase(_LayerBase):
    def __init__(self):
        super(_ActivationLayerBase, self).__init__(self._name, types.ACTIVATION_LAYER)


class SigmoidLayer(_ActivationLayerBase):
    """
    This layer takes all inputs and transforms them by the function listed below.
    This transformation creates the non-linearity in a neural network.
    The input is typically the output of convolution, pooling, or a fully connected layer.
    You cannot have two activation layers in a row.

    .. math::

        f(x) = 1/(1+e^{-x})


    See Also
    --------
    RectifiedLinearLayer, TanhLayer, SoftplusLayer

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.SigmoidLayer()

    """
    _name = "sigmoid"


class RectifiedLinearLayer(_ActivationLayerBase):
    """
    This layer takes all inputs and transforms them by the function listed below.
    This transformation creates the non-linearity in a neural network.
    The input is typically the output of convolution, pooling, or a fully connected layer.
    You cannot have two activation layers in a row.


    .. math::

        f(x) = max(0, x)

    See Also
    --------
    SigmoidLayer,TanhLayer,SoftplusLayer

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.RectifiedLinearLayer()

    """
    _name = "relu"


class TanhLayer(_ActivationLayerBase):
    """
    This layer takes all inputs and transforms them by the function listed below.
    This transformation creates the non-linearity in a neural network.
    The input is typically the output of convolution, pooling, or a fully connected layer.
    You cannot have two activation layers in a row.


    .. math::

        f(x) = tanh(x)

    See Also
    --------
    SigmoidLayer, RectifiedLinearLayer, SoftplusLayer

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.TanhLayer()

    """
    _name = "tanh"


class SoftplusLayer(_ActivationLayerBase):
    """
    This layer takes all inputs and transforms them by the function listed below.
    This transformation creates the non-linearity in a neural network.
    The input is typically the output of convolution, pooling, or a fully connected layer.
    You cannot have two activation layers in a row.


    .. math::

         f(x) = log(1+e^x)

    See Also
    --------
    SigmoidLayer, RectifiedLinearLayer, TanhLayer

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.SoftplusLayer()

   """
    _name = "softplus"


# /**************************************************************************/
# /*                                                                        */
# /*                           Convolution Layers                           */
# /*                                                                        */
# /**************************************************************************/
class ConvolutionLayer(_LayerBase):
    """
    A connection layer which convolves a learned linear filter over the input,
    creating a feature map. There are 'num_channel' filters, creating
    'num_channel' feature maps. See References for more information.

    Parameters
    ----------
    kernel_size : int
        The size of one side of a square filter for convolution.
    num_channels : int
        The number of feature maps or filters learned.
    stride : int, optional. Default 1.
        The intervals at which to apply kernels to the input.
    padding : int, optional. Default 0.
        Amount of zero padding on each side of the input.
    num_groups : int, optional. Default 1.
        If this is greater than 1, each filter is restricted to a subset of
        the input. Input and output maps are divided into 'num_groups'
        groups, and one input group is mapped to one output group.
    init_random: {'gaussian', 'xavier'}, optional. Default 'gaussian'.
        The distribution from which to initialize parameters:
        Gaussian is a normal distribution
        Xavier determines the scale of initialization based on number
        of input and output neurons.


    References
    ----------
    - `Tutorial on Convolutional Neural Networks <http://deeplearning.net/tutorial/lenet.html#lenet>`_

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.ConvolutionLayer(kernel_size=3, num_channels=10)

    """
    def __init__(self, kernel_size, num_channels, stride=1, padding=0, num_groups=1, init_random='gaussian'):
        super(ConvolutionLayer, self).__init__("conv", types.CONVOLUTION_LAYER)
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.init_random = init_random

    def _get_output_shape(self, input_shape):
        return [(input_shape[0] + 2 * self.padding - self.kernel_size) / self.stride + 1,
                (input_shape[1] + 2 * self.padding - self.kernel_size) / self.stride + 1,
                self.num_channels]

    def _verify(self):
        if (self.kernel_size <= 0):
            raise ValueError('kernel_size must be positive')
        if (self.num_channels <= 0):
            raise ValueError('num_channels must be positive')
        if (self.padding < 0):
            raise ValueError('padding must be positive')
        if (self.stride <= 0):
            raise ValueError('stride must be positive')
        if (self.num_groups <= 0):
            raise ValueError('num_groups must be positive')
        return super(ConvolutionLayer, self)._verify()


# /**************************************************************************/
# /*                                                                        */
# /*                             Pooling Layers                             */
# /*                                                                        */
# /**************************************************************************/
class _PoolingLayerBase(_LayerBase):
    def __init__(self, kernel_size, stride=1, padding=0):
        if (kernel_size <= 0):
            raise ValueError('kernel_size must be positive')
        super(_PoolingLayerBase, self).__init__(self._name, types.POOLING_LAYER)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


    def _get_output_shape(self, input_shape):
        return [(input_shape[0] + 2 * self.padding - self.kernel_size) / self.stride + 1, (input_shape[1] + 2 * self.padding - self.kernel_size) / self.stride + 1, input_shape[2]]

    def _verify(self):
        if (self.kernel_size <= 0):
            raise ValueError('kernel_size must be positive')
        if (self.padding < 0):
            raise ValueError('padding must be positive')
        if (self.stride <= 0):
            raise ValueError('stride must be positive')
        return super(_PoolingLayerBase, self)._verify()


class MaxPoolingLayer(_PoolingLayerBase):
    """
    Max-Pooling partitions the input rectangles denoted by the kernel size.
    For each rectangle, it outputs the maximum value. This is a form
    of downsampling. See References for more information.

    Parameters
    ----------
    kernel_size : int
        The size of one side of a square filter for pooling.
    stride : int, optional. Default 1.
        The intervals at which to apply pooling  to the input.
    padding : int , optional. Default 0.
        Amount of zero padding on each side of the input.

    See Also
    --------
    SumPoolingLayer, AveragePoolingLayer

    References
    ----------
    - `Tutorial on Pooling <http://ufldl.stanford.edu/wiki/index.php/Pooling>`_

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.MaxPoolingLayer(kernel_size=3)

    """
    _name = 'max_pooling'


class SumPoolingLayer(_PoolingLayerBase):
    """
    Sum-Pooling partitions the input rectangles denoted by the kernel size.
    For each rectangle, it outputs the sum of each value in the rectangle.
    This is a form of downsampling.  See References for more information.

    Parameters
    ----------
    kernel_size : int
        The size of one side of a square filter for pooling.
    stride : int, optional. Default 1.
        The intervals at which to apply pooling  to the input.
    padding : int , optional. Default 0.
        Amount of zero padding on each side of the input.

    See Also
    --------
    AveragePoolingLayer, MaxPoolingLayer

    References
    ----------
    - `Tutorial on Pooling <http://ufldl.stanford.edu/wiki/index.php/Pooling>`_

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.SumPoolingLayer(kernel_size=3)

    """

    _name = 'sum_pooling'


class AveragePoolingLayer(_PoolingLayerBase):
    """
    Average-Pooling partitions the input rectangles denoted by the kernel size.
    For each rectangle, it outputs the average of each value in the rectangle.
    This is a form of downsampling.  See References for more information.

    Parameters
    ----------
    kernel_size : int
        The size of one side of a square filter for pooling.
    stride : int, optional. Default 1.
        The intervals at which to apply pooling  to the input.
    padding : int , optional. Default 0.
        Amount of zero padding on each side of the input.

    See Also
    --------
    SumPoolingLayer, MaxPoolingLayer

    References
    ----------
    - `Tutorial on Pooling <http://ufldl.stanford.edu/wiki/index.php/Pooling>`_

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.AveragePoolingLayer(kernel_size=3)

    """
    _name = 'avg_pooling'


# /**************************************************************************/
# /*                                                                        */
# /*                         Regularization Layers                          */
# /*                                                                        */
# /**************************************************************************/
class DropoutLayer(_LayerBase):
    """
    A dropout layer takes all inputs, and with probability 'threshold', sets it
    to 0. This prevents the network from overfitting to the data.

    Parameters
    ----------
    threshold : float, optional. Default 0.5
        The probability with which to set each input unit to 0.

    See Also
    --------
    LocalResponseNormalizationLayer

    References
    ----------
    - Hinton, Geoffrey E., et al. "Improving neural networks by preventing
      co-adaptation of feature detectors." arXiv preprint arXiv:1207.0580
      (2012).

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.DropoutLayer()

    """
    def __init__(self, threshold=0.5):
        super(DropoutLayer, self).__init__("dropout", types.REGULARIZATION_LAYER)
        self.threshold = threshold

    def _verify(self):
        if (self.threshold > 1 or self.threshold < 0):
            raise ValueError('threshold hold must be between [0, 1]')


class LocalResponseNormalizationLayer(_LayerBase):
    """
    Local response normalization normalizes output values by the values
    present in neighboring maps in corresponding positions. Must have non-flat input.
    Each input value is divided by :math:`(knorm + \\alpha*\sum(x_i^2))^\\beta` for :math:`x_i`'s in the
    local neighberhood of size 'local_size' centered at the input value.

    Parameters
    ----------
    alpha : float, optional. Default 0.001.
        A hyperparamter in the normalization equation, increassing
        increases the regularization. Default value of 0.001 is
        recommended.
    beta : float , optional. Default 0.75.
        A hyperparameter in the normalization equation, increasing increases
        the regularization.
    local_size : int , optional. Default 5.
        A hyperparamter in the normalization equation, sets the size
        of the normalization neighborhood.
    knorm : float , optional. Default 1.
        A hyperparameter in the normalization equation, sets the bias
        of the normalization.

    See Also
    --------
    DropoutLayer

    References
    ----------
    - Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
      classification with deep convolutional neural networks." Advances in
      neural information processing systems. 2012.

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.LocalResponseNormalizationLayer()

    """

    def __init__(self, alpha=0.001, beta=0.75, local_size=5, knorm=1):
        super(LocalResponseNormalizationLayer, self).__init__("lrn", types.REGULARIZATION_LAYER)
        self.alpha = alpha
        self.beta = beta
        self.local_size = local_size
        self.knorm = knorm

    def _verify(self):
        if (self.alpha <= 0):
            raise ValueError('alpha must be positive')
        if (self.beta <= 0):
            raise ValueError('beta must be positive')
        if (self.local_size <= 0 or self.local_size % 2 == 0):
            raise ValueError('local_size must be positive and odd')
        if (self.knorm <= 0):
            raise ValueError('knorm must be positive')
        return super(LocalResponseNormalizationLayer, self)._verify()


# /**************************************************************************/
# /*                                                                        */
# /*                             Output Layers                              */
# /*                                                                        */
# /**************************************************************************/
class SoftmaxLayer(_LayerBase):
    """
    Output layer for multiclass classification output.

    Takes each input unit, and transforms it by the function:

    .. math::

        f(x_i) = e^{x_i}/\sum_{k}e^{x_k}

    where :math:`x_i` is the :math:`i` th output and the summation is over all
    :math:`k` inputs.

    In this case, each output unit corresponds to a training class. The output value
    of each unit is the learned probability that the input to the network is
    part of the class corresponding to that unit.

    The loss associated with the softmax layer is the multinomial logistic loss.

    Previous layer must be fully connected layer, with number of hidden units
    equal to number of classes to be classified.

    References
    ----------
    - `Tutorial on Softmax Regression <http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/>`_

    Examples
    --------

    >>> layer = graphlab.deeplearning.layers.SoftmaxLayer()

    """
    def __init__(self):
        super(SoftmaxLayer, self).__init__("softmax", types.OUTPUT_LAYER)


def _layer_from_string(s):
    name_to_class_map = {'fullc': FullConnectionLayer,
                         'flatten': FlattenLayer,
                         'conv': ConvolutionLayer,
                         'sigmoid': SigmoidLayer,
                         'tanh': TanhLayer,
                         'relu': RectifiedLinearLayer,
                         'softplus': SoftplusLayer,
                         'max_pooling': MaxPoolingLayer,
                         'sum_pooling': SumPoolingLayer,
                         'avg_pooling': AveragePoolingLayer,
                         'lrn': LocalResponseNormalizationLayer,
                         'dropout': DropoutLayer,
                         'softmax': SoftmaxLayer}

    cls = name_to_class_map[s]
    if s in ('max_pooling', 'sum_pooling', 'avg_pooling'):
        return cls(kernel_size=2)
    elif s == 'conv':
        return cls(kernel_size=2, num_channels=10)
    elif s == 'fullc':
        return cls(num_hidden_units=20)
    else:
        return cls()
