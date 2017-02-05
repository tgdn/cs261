from .. import symbol

def get_symbol(num_classes = 1000):
    """
    Return the "VGG" architecture for image classification

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the ouptut layer.

    References
    ----------
    Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
    large-scale image recognition." arXiv preprint arXiv:1409.1556, 2014.
    """

    ## define alexnet
    data = symbol.Variable(name="data")
    # group 1
    conv1_1 = symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = symbol.Pooling(
        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = symbol.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = symbol.Flatten(data=pool5, name="flatten")
    fc6 = symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    fc8 = symbol.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    softmax = symbol.SoftmaxOutput(data=fc8, name='softmax')
    return softmax
