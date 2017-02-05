from .. import symbol

# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
    conv = symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = symbol.BatchNorm(data=conv)
    act = symbol.Activation(data = bn, act_type=act_type)
    return act

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1))
    # pool
    pool = symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max')
    # concat
    concat = symbol.Concat(*[conv, pool])
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3)
    #concat
    concat = symbol.Concat(*[conv1x1, conv3x3])
    return concat

def get_symbol(num_classes = 10):
    """
    Return a simplified version of "BN-Inception" architecture for image classification

    The network is suitable for images of size around 28 x 28

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the ouptut layer.

    References
    ----------
    - Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep
      network training by reducing internal covariate shift. arXiv preprint
      arXiv:1502.03167, 2015.
    """
    data = symbol.Variable(name="data")
    conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu")
    in3a = SimpleFactory(conv1, 32, 32)
    in3b = SimpleFactory(in3a, 32, 48)
    in3c = DownsampleFactory(in3b, 80)
    in4a = SimpleFactory(in3c, 112, 48)
    in4b = SimpleFactory(in4a, 96, 64)
    in4c = SimpleFactory(in4b, 80, 80)
    in4d = SimpleFactory(in4c, 48, 96)
    in4e = DownsampleFactory(in4d, 96)
    in5a = SimpleFactory(in4e, 176, 160)
    in5b = SimpleFactory(in5a, 176, 160)
    pool = symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool")
    flatten = symbol.Flatten(data=pool, name="flatten1")
    fc = symbol.FullyConnected(data=flatten, num_hidden=num_classes, name="fc1")
    softmax = symbol.SoftmaxOutput(data=fc, name="softmax")
    return softmax
