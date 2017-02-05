from .. import symbol

def get_symbol(num_classes = 1000):
    """
    Return the "AlexNet" architecture for image classification

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the ouptut layer.

    References
    ----------
    - Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
      classification with deep convolutional neural networks." Advances in neural
      information processing systems. 2012.
    """
    input_data = symbol.Variable(name="data")
    # stage 1
    conv1 = symbol.Convolution(
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = symbol.Activation(data=conv1, act_type="relu")
    pool1 = symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    lrn1 = symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = symbol.Convolution(
        data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = symbol.Activation(data=conv2, act_type="relu")
    pool2 = symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = symbol.Convolution(
        data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = symbol.Activation(data=conv3, act_type="relu")
    conv4 = symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = symbol.Activation(data=conv4, act_type="relu")
    conv5 = symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = symbol.Activation(data=conv5, act_type="relu")
    pool3 = symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = symbol.Flatten(data=pool3)
    fc1 = symbol.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = symbol.Activation(data=fc1, act_type="relu")
    dropout1 = symbol.Dropout(data=relu6, p=0.5)
    # stage 5
    fc2 = symbol.FullyConnected(data=dropout1, num_hidden=4096)
    relu7 = symbol.Activation(data=fc2, act_type="relu")
    dropout2 = symbol.Dropout(data=relu7, p=0.5)
    # stage 6
    fc3 = symbol.FullyConnected(data=dropout2, num_hidden=num_classes)
    softmax = symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax
