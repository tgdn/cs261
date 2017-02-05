"""
The deeplearning module is useful to create and manipulate
different neural network architectures. The core of this module
is the :py:class:`~graphlab.deeplearning.NeuralNet` class,
which stores the definition of
each layer of a neural network and a dictionary of learning parameters.

A :py:class:`~graphlab.deeplearning.NeuralNet` object can be obtained from
:py:func:`deeplearning.create() <graphlab.deeplearning.create>`.
The returned network can be provided to :func:`graphlab.neuralnet_classifier.create`
to train a :class:`~graphlab.neuralnet_classifier.NeuralNetClassifier`.

The general workflow involves three steps:

1. Create a neuralnet using :func:`graphlab.deeplearning.create`
2. Tweak the neuralnet parameters so that they are appropriate for the
   desired task
3. Pass the net to :func:`graphlab.neuralnet_classifier.create` to train
   the classifier with the given network architechture and parameters.

This function selects a *default*
network architecture depending on the input dataset using simple rules: it
creates a 1-layer Perceptron Network for dense numeric input, and a 1-layer
Convolution Network for image data input.

Due to the high complexity of
netualnet models, the default network does not always work out of the box, and
you will often need to tweak the architectures a bit to make it work for your
problem. Please see the reference section for a practical guide of tuning
deeplearning architectures.


**Example: Digit Recognition on MNIST Data**

Training a convolutional neural network for digits recognition.

>>> import graphlab

Load the MNIST data. We have downloaded the dataset
from the MNIST database, and save into SFrames on S3.

>>> data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/train')
>>> test_data = graphlab.SFrame('https://static.turi.com/datasets/mnist/sframe/test')
>>> training_data, validation_data = data.random_split(0.8)

We need to make sure all of the images are the same size, since neural nets have
fixed input size.

>>> training_data['image'] = graphlab.image_analysis.resize(training_data['image'], 28, 28, 1, decode=True)
>>> validation_data['image'] = graphlab.image_analysis.resize(validation_data['image'], 28, 28, 1, decode=True)
>>> test_data['image'] = graphlab.image_analysis.resize(test_data['image'], 28, 28, 1, decode=True)


Use the builtin NeuralNet architecture for MNIST (
a one layer convolutional neuralnet work)

>>> net = graphlab.deeplearning.create(training_data, target='label')

Layers of the neural network

>>> net.layers
layer[0]: ConvolutionLayer
  stride = 2
  num_channels = 10
  kernel_size = 3
layer[1]: RectifiedLinearLayer
layer[2]: MaxPoolingLayer
  stride = 2
  kernel_size = 3
layer[3]: FlattenLayer
layer[4]: FullConnectionLayer
  num_hidden_units = 100
layer[5]: RectifiedLinearLayer
layer[6]: DropoutLayer
  threshold = 0.5
layer[7]: FullConnectionLayer
  num_hidden_units = 10
layer[8]: SoftmaxLayer

Parameters of the neural network

>>> net.params
{'learning_rate': 0.001, 'momentum': 0.9}

Tweak some hyperparameters

>>> net.layers[4].num_hidden_units = 10

Train a NeuralNetClassifier using the specified network.

>>> m = graphlab.neuralnet_classifier.create(training_data, target='label',
...                                          network = net,
...                                          validation_set=validation_data,
...                                          metric=['accuracy', 'recall@2'],
...                                          max_iterations=3)

Classify the test data, and output the most likely class label. The score
corresponds to the learned probability of the testing instance belonging
to the predicited class.

>>> pred = m.classify(test_data)
>>> pred
+--------+-------+----------------+
| row_id | class |     score      |
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


Predict the top 2 most likely digits

>>> pred_top2 = m.predict_topk(test_data, k=2)
>>> pred_top2
+--------+-------+-------------------+
| row_id | class |       score       |
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

Evaluate the classifier on the test data. Default metrics are
accuracy, and confusion matrix.

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

**Builtin NeuralNets**

There are two builtin example architectures for
reproducing the MNIST and ImageNet experiments. In References, see *'Imagenet Classification
with Deep Convolutional Neural Networks'* for ImageNet architecture explanation
and *'Gradient-based Learning Applied to Document Recognition'* for MNIST
architecture explanation:

>>> mnist_net = graphlab.deeplearning.get_builtin_neuralnet('mnist')

>>> imagenet_net = graphlab.deeplearning.get_builtin_neuralnet('imagenet')

**Create NeuralNet from Data**

It is possible to create a network from a dataset. GraphLab Create intelligently chooses
a network architecture based on input data shape. If the input is of image type, a
convolutional neural network is chosen. Otherwise, a simple multi-layer
perceptron network is chosen.

.. sourcecode:: python

    # Import graphlab and load the MNIST dataset.
    >>> import graphlab as gl
    >>> dataset = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train6k')

    # Create a default NeuralNet for MNSIT data.
    >>> net = gl.deeplearning.create(dataset, target='label')

    # The architecture is represented as a list of `Layer` objects.
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

    # And the learning parameter is stored as a dictionary.
    >>> net.params
    {'learning_rate': 0.001, 'momentum': 0.9}

    # The NeuralNet object has a human readable string representation.
    >>> net
    ### network layers ###
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
    ### end network layers ###
    <BLANKLINE>
    ### network parameters ###
    learning_rate = 0.001
    momentum = 0.9
    ### end network parameters ###


**Create a NeuralNet from Templates**

Most of the time, it is easy to start with a default NeuralNet, and tweak the
layers and parameters to fit your problem. However, sometimes it is desirable
to create a NeuralNet from scratch. We provide two templated class to
make it easy: :py:class:`~graphlab.deeplearning.MultiLayerPerceptrons`,
and :py:class:`~graphlab.deeplearning.ConvolutionNet`.

Create a MultiLayerPerceptrons network with 3 hidden layers, the last layer
is used for output.

>>> percpt_net = gl.deeplearning.MultiLayerPerceptrons(num_hidden_layers=3,
                                                num_hidden_units=[15,10,10])

Create a ConvolutionNet with 2 convolution layers and 1 output layer
with 10 units.

>>> conv_net = gl.deeplearning.ConvolutionNet(num_convolution_layers=2,
                                              kernel_size=3,
                                              num_channels=10,
                                              num_output_units=10)

Stack the 2-layer perceptron net on top of the convolution net.We re-create
the ConvolutionNet so that output size is 0, so we can define an output layer of
our own.

>>> conv_net = gl.deeplearning.ConvolutionNet(num_convolution_layers=2,
                                              kernel_size=3,
                                              num_channels=10,
                                              num_output_units=0)

>>> conv_net.layers.extend(percpt_net.layers)
>>> conv_net.layers
layer[0]: ConvolutionLayer
  stride = 1
  num_channels = 10
  kernel_size = 3
layer[1]: RectifiedLinearLayer
layer[2]: MaxPoolingLayer
  stride = 1
  kernel_size = 3
layer[3]: ConvolutionLayer
  stride = 1
  num_channels = 10
  kernel_size = 3
layer[4]: RectifiedLinearLayer
layer[5]: MaxPoolingLayer
  stride = 1
  kernel_size = 3
layer[6]: FlattenLayer
layer[7]: FullConnectionLayer
  num_hidden_units = 15
layer[8]: SigmoidLayer
layer[9]: FullConnectionLayer
  num_hidden_units = 10
layer[10]: SigmoidLayer
layer[11]: FullConnectionLayer
  num_hidden_units = 10
layer[12]: SoftmaxLayer

Verify that the layers form a valid network architecture.

>>> conv_net.verify()
True

**Modify NeuralNet**

The NeuralNet object is used to create a :py:class:`~graphlab.neuralnet_classifier.NeuralNetClassifier`.

>>> model = gl.neuralnet_classifier.create(dataset, target='label', network=net)

To change the a specific layer, simply modify its attribute.

>>> net.layers[-2].num_hidden_units=2

To add or remove an layer, simply use the python list operation.

>>> del net.layers[-1]
>>> net.layers.append(gl.deeplearning.layers.SoftmaxLayer())

To add parameters to the Neuralnet, simply use the python dict operation.

>>> net.params['learning_rate'] = 0.1

**Save and load NeuralNet**

NeuralNet object can be saved into a text file using
:py:func:`~graphlab.neuralnet_classifier.NeuralNetClassifier.save()`.

>>> net = gl.deeplearning.create(dataset, target='label')
>>> net.save('mynet')

It can be loaded back using :py:func:`~graphlab.deeplearning.load()`.

>>> net2 = gl.deeplearning.load('mynet')

Use :py:func:`~graphlab.deeplearning.loads()` to load a network from string.

>>> net_string = file('mynet').read()
>>> net2 = gl.deeplearning.loads(net_string)

**See Also**

:py:func:`graphlab.neuralnet_classifier.create`
:py:class:`~graphlab.neuralnet_classifier.NeuralNetClassifier`

**References**

- Bengio, Yoshua. "Practical recommendations for gradient-based training of
  deep architectures." Neural Networks: Tricks of the Trade. Springer Berlin
  Heidelberg, 2012. 437-478.

- LeCun, Yann, et al. "Gradient-based learning applied to document
  recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

- `The MNIST database <http://yann.lecun.com/exdb/mnist/>`_

- Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification
  with deep convolutional neural networks." Advances in neural information
  processing systems. 2012.


"""


from . _neuralnets import NeuralNet, MultiLayerPerceptrons, ConvolutionNet
from . import layers
from ._main import create, load, loads, get_builtin_neuralnet
