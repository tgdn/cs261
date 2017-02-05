# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments, W0221, W0201, no-self-use

"""NDArray interface of mxnet"""
from __future__ import absolute_import
from collections import OrderedDict

import ctypes
import sys
import numpy as np
import logging
import threading
from .base import _LIB
from .base import c_array, c_str, mx_uint, py_str
from .base import DataIterHandle, NDArrayHandle
from .base import check_call, ctypes2docstring
from array import array as _array
from .ndarray import NDArray
from .ndarray import array
from .ndarray import _copy_from_sarray, _copy_from_sframe

class DataBatch(object):
    """Default object for holding a mini-batch of data and related information."""
    def __init__(self, data, label, pad, index,
                 bucket_key=None, provide_data=None, provide_label=None):
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index

class DataBatch(object):
    """Default object for holding a mini-batch of data and related information."""
    def __init__(self, data, label, pad=None, index=None,
                 bucket_key=None, provide_data=None, provide_label=None):
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index

        # the following properties are only used when bucketing is used
        self.bucket_key = bucket_key
        self.provide_data = provide_data
        self.provide_label = provide_label

class DataIter(object):
    """DataIter object in mxnet. """

    def __init__(self):
        self.batch_size = 0

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator. """
        pass

    def next(self):
        """Get next data batch from iterator.

        Equivalent to

        .. code-block:: python

            self.iter_next()
            DataBatch(self.getdata(), self.getlabel(), self.getpad(), None)

        Returns
        -------
        data : DataBatch
            The data of next batch.
        """
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Iterate to next batch.

        Returns
        -------
        has_next : boolean
            Whether the move is successful.
        """
        pass

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """
        pass

    def getlabel(self):
        """Get label of current batch.

        Returns
        -------
        label : NDArray
            The label of current batch.
        """
        pass

    def getindex(self):
        """Get index of the current batch.

        Returns
        -------
        index : numpy.array
            The index of current batch
        """
        return None

    def getpad(self):
        """Get the number of padding examples in current batch.

        Returns
        -------
        pad : int
            Number of padding examples in current batch
        """
        pass

class ResizeIter(DataIter):
    """Resize a DataIter to given number of batches per epoch.
    May produce incomplete batch in the middle of an epoch due
    to padding from internal iterator.

    Parameters
    ----------
    data_iter : DataIter
        Internal data iterator.
    size : number of batches per epoch to resize to.
    reset_internal : whether to reset internal iterator on ResizeIter.reset
    """

    def __init__(self, data_iter, size, reset_internal=True):
        super(ResizeIter, self).__init__()
        self.data_iter = data_iter
        self.size = size
        self.reset_internal = reset_internal
        self.cur = 0
        self.current_batch = None

        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size

    def reset(self):
        self.cur = 0
        if self.reset_internal:
            self.data_iter.reset()

    def iter_next(self):
        if self.cur == self.size:
            return False
        try:
            self.current_batch = self.data_iter.next()
        except StopIteration:
            self.data_iter.reset()
            self.current_batch = self.data_iter.next()

        self.cur += 1
        return True

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

class PrefetchingIter(DataIter):
    """Base class for prefetching iterators. Takes one or more DataIters (
    or any class with "reset" and "read" methods) and combine them with

    Parameters
    ----------
    iters : DataIter or list of DataIter
        one or more DataIters (or any class with "reset" and "read" methods)
    rename_data : None or list of dict
        i-th element is a renaming map for i-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data
    rename_label : None or list of dict
        Similar to rename_data

    Examples
    --------
    >>> iter = PrefetchingIter([NDArrayIter({'data': X1}), NDArrayIter({'data': X2})],
    ...                        rename_data=[{'data': 'data1'}, {'data': 'data2'}])
    """
    def __init__(self, iters, rename_data=None, rename_label=None):
        super(PrefetchingIter, self).__init__()
        if not isinstance(iters, list):
            iters = [iters]
        self.n_iter = len(iters)
        assert self.n_iter > 0
        self.iters = iters
        if rename_data is None:
            self.provide_data = sum([i.provide_data for i in iters], [])
        else:
            self.provide_data = sum([[(r[n], s) for n, s in i.provide_data] \
                                    for r, i in zip(rename_data, iters)], [])
        if rename_label is None:
            self.provide_label = sum([i.provide_label for i in iters], [])
        else:
            self.provide_label = sum([[(r[n], s) for n, s in i.provide_label] \
                                    for r, i in zip(rename_label, iters)], [])
        self.batch_size = self.provide_data[0][1][0]
        self.data_ready = [threading.Event() for i in range(self.n_iter)]
        self.data_taken = [threading.Event() for i in range(self.n_iter)]
        for e in self.data_taken:
            e.set()
        self.started = True
        self.current_batch = [None for i in range(self.n_iter)]
        self.next_batch = [None for i in range(self.n_iter)]
        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.next_batch[i] = self.iters[i].next()
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()
        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                                 for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()

    def reset(self):
        for e in self.data_ready:
            e.wait()
        for i in self.iters:
            i.reset()
        for e in self.data_ready:
            e.clear()
        for e in self.data_taken:
            e.set()

    def iter_next(self):
        for e in self.data_ready:
            e.wait()
        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            for batch in self.next_batch:
                assert batch.pad == self.next_batch[0].pad, \
                    "Number of entry mismatches between iterators"
            self.current_batch = DataBatch(sum([batch.data for batch in self.next_batch], []),
                                           sum([batch.label for batch in self.next_batch], []),
                                           self.next_batch[0].pad,
                                           self.next_batch[0].index)
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

def _init_data(data, allow_empty, default_name):
    """Convert data into canonical form."""
    assert (data is not None) or allow_empty
    if data is None:
        data = []

    if isinstance(data, (np.ndarray, NDArray)):
        data = [data]
    if isinstance(data, list):
        if not allow_empty:
            assert(len(data) > 0)
        if len(data) == 1:
            data = OrderedDict([(default_name, data[0])])
        else:
            data = OrderedDict([('_%d_%s' % (i, default_name), d) for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError("Input must be NDArray, numpy.ndarray, " + \
                "a list of them or dict with them as values")
    for k, v in data.items():
        if isinstance(v, NDArray):
            data[k] = v.asnumpy()
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(("Invalid type '%s' for %s, "  % (type(v), k)) + \
                    "should be NDArray or numpy.ndarray")

    return list(data.items())

class NDArrayIter(DataIter):
    """NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.
    Parameters
    ----------
    data: NDArray or numpy.ndarray, a list of them, or a dict of string to them.
        NDArrayIter supports single or multiple data and label.
    label: NDArray or numpy.ndarray, a list of them, or a dict of them.
        Same as data, but is not fed to the model during testing.
    batch_size: int
        Batch Size
    shuffle: bool
        Whether to shuffle the data
    last_batch_handle: 'pad', 'discard' or 'roll_over'
        How to handle the last batch

    Notes
    -----
    This iterator will pad, discard or roll over the last batch if
    the size of data does not match batch_size. Roll over is intended
    for training and can cause problems if used for prediction.
    """
    def __init__(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad'):
        # pylint: disable=W0201

        super(NDArrayIter, self).__init__()

        self.data = _init_data(data, allow_empty=False, default_name='data')
        self.label = _init_data(label, allow_empty=True, default_name='softmax_label')

        # shuffle data
        if shuffle:
            idx = np.arange(self.data[0][1].shape[0])
            np.random.shuffle(idx)
            self.data = [(k, v[idx]) for k, v in self.data]
            self.label = [(k, v[idx]) for k, v in self.label]

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data_list[0].shape[0] - self.data_list[0].shape[0] % batch_size
            data_dict = OrderedDict(self.data)
            label_dict = OrderedDict(self.label)
            for k, _ in self.data:
                data_dict[k] = data_dict[k][:new_n]
            for k, _ in self.label:
                label_dict[k] = label_dict[k][:new_n]
            self.data = data_dict.items()
            self.label = label_dict.items()
        self.num_data = self.data_list[0].shape[0]
        assert self.num_data >= batch_size, \
            "batch_size need to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]


    def hard_reset(self):
        """Igore roll over data and set to start"""
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return [array(x[1][self.cursor:self.cursor+self.batch_size]) for x in data_source]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [array(np.concatenate((x[1][self.cursor:], x[1][:pad]),
                                         axis=0)) for x in data_source]

    def getdata(self):
        return self._getdata(self.data)

    def getlabel(self):
        return self._getdata(self.label)

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0

try:
    import graphlab as gl
except:
    try:
        import sframe as gl
    except:
        pass


class SFrameIter(DataIter):
    """DataIter from SFrame
    Provides DataIter interface for SFrame, a highly scalable columnar DataFrame.
    The iterator can simultaneously iterate over multiple columns indicated by `data_field` and `label_field`.
    `data_field` can refer either a single image typed column or multiple numerical columns (int, float or array).
    `label_field` con only refer to a single numerical column (int, float or array).

    Parameters
    ----------
    sframe : SFrame object
        source SFrame
    data_field : string or list(string)
        data fields of the SFrame. The seleted fields may be either a single image typed column,
        or multiple numerical columns (int, float, array).
    label_field : string, optional
        label field in SFrame
    batch_size : int, optional
        batch size

    Examples
    --------
    >>> import sframe as sf
    >>> import mxnet as mx

    >>> data = sf.SFrame({'x': [1,2,3], 'y': [.1, .5, .5], 'z': [[1,1,1], [2,2,2,], [3,3,3]]})
    >>> dataiter = mx.io.SFrameIter(sframe=data, data_field=['x', 'z'], label_field='z')

    >>> image_data = sf.SFrame('http://s3.amazonaws.com/dato-datasets/mnist/sframe/train')
    >>> image_data_iter = mx.io.SFrameIter(sframe=data, data_field=['image'], label_field='label', batch_size=100)

    Notes
    -----
    - Image column must contain images of the same size.
    - Array column must contain arrays of the same length.
    """

    def __init__(self, sframe, data_field, label_field=None, batch_size=1, data_name='data', label_name='softmax_label'):

        super(SFrameIter, self).__init__()
        if not isinstance(sframe, gl.SFrame):
            raise TypeError
        if not (isinstance(data_field, str) or isinstance(data_field, list)):
            raise TypeError
        if not (label_field is None or isinstance(label_field, str)):
            raise TypeError

        if type(data_field) is str:
            data_field = [data_field]

        self._type_check(sframe, data_field, label_field)
        self.data_field = data_field
        self.label_field = label_field
        self.data_sframe = sframe[data_field]
        if label_field is not None:
            self.label_sframe = sframe[label_field]

        # allocate ndarray
        inferred_shape = self.infer_shape()
        data_shape = list(inferred_shape["final_shape"])
        data_shape.insert(0, batch_size)
        self.data_shape = tuple(data_shape)
        self.label_shape = (batch_size, )
        self.field_length = inferred_shape["field_length"]
        self.data_ndarray = array(np.zeros(self.data_shape))
        self.label_ndarray = array(np.zeros(self.label_shape))
        self.data = _init_data(self.data_ndarray, allow_empty=False, default_name=data_name)
        self.label = _init_data(self.label_ndarray, allow_empty=True, default_name=label_name)
        # size
        self.batch_size = batch_size
        self.data_size = len(sframe)
        self.reset()

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]

    def reset(self):
        self.pad = 0
        self.cursor = 0
        self.has_next = True

    def _type_check(self, sframe, data_field, label_field):
        if label_field is not None:
            label_column_type = sframe[label_field].dtype()
            if label_column_type not in [int, float]:
                raise TypeError('Unexpected type for label_field \"%s\". Expect int or float, got %s' %
                                (label_field, str(label_column_type)))
        for col in data_field:
            col_type = sframe[col].dtype()
            if col_type not in [int, float, _array, gl.Image]:
                raise TypeError('Unexpected type for data_field \"%s\". Expect int, float, array or image, got %s' %
                               (col, str(col_type)))

    def _infer_column_shape(self, sarray):
        dtype = sarray.dtype()
        if (dtype in [int, float]):
            return (1, )
        elif dtype is _array:
            lengths = sarray.item_length()
            if lengths.min() != lengths.max():
                raise ValueError('Array column does not have the same length')
            else:
                return (lengths.max(), )
        elif dtype is gl.Image:
            first_image = sarray.head(1)[0]
            if first_image is None:
                raise ValueError('Column cannot contain missing value')
            return (first_image.channels, first_image.height, first_image.width)

    def infer_shape(self):
        ret = {"field_length": [], "final_shape": None}
        features = self.data_sframe.column_names()
        assert len(features) > 0
        if len(features) > 1:
            # If more than one feature, all features must be numeric or array
            shape = 0
            for col in features:
                colshape = self._infer_column_shape(self.data_sframe[col])
                if len(colshape) != 1:
                    raise ValueError('Only one column is allowed if input is image typed')
                shape += colshape[0]
                ret["field_length"].append(colshape[0])
            ret["final_shape"] = (shape,)
        else:
            col_shape = self._infer_column_shape(self.data_sframe[features[0]])
            ret["final_shape"] = col_shape
            length = 1
            for x in col_shape:
                length = length * x
            ret["field_length"].append(length)
        return ret

    def _copy(self, start, end, bias=0):
        _copy_from_sframe(self.data_sframe, self.data_ndarray, start, end, self.field_length, bias)
        if self.label_field is not None:
            _copy_from_sarray(self.label_sframe, self.label_ndarray, start, end, 1, bias)

    def iter_next(self):
        if self.has_next:
            start = self.cursor
            end = start + self.batch_size
            if end >= self.data_size:
                self.has_next = False
                self.pad = end - self.data_size
                end = self.data_size
            self._copy(start, end)
            if self.pad > 0:
                bias = self.batch_size - self.pad
                start = 0
                end = self.pad
                self._copy(start, end, bias)
                self.cursor = self.pad
            else:
                self.cursor += self.batch_size
            return True
        else:
            return False

    def getdata(self):
        return [self.data_ndarray]

    def getlabel(self):
        return [self.label_ndarray]

    def getpad(self):
        return self.pad


class SFrameImageIter(SFrameIter):
    """Image Data Iterator from SFrame
    Provide the SFrameIter like interface with options to normalize and augment image data.

    Parameters
    ----------
    sframe : SFrame object
        source SFrame
    data_field : string
        image data field of the SFrame.
    label_field : string, optional
        label field in SFrame
    batch_size : int, optional
        batch size
    mean_r : float, optional
        normalize the image by subtracting the mean value of r channel, or the first channel for
    mean_g : float, optional
        normalize the image by subtracting the mean value of g channel
    mean_b : float, optional
        normalize the image by subtracting the mean value of b channel
    mean_nd : np.ndarray, optional
        normalize the image by subtracting the ndarray of mean pixel values.
        The mean_nd array stores the pixel values in the order of [height, width, channel]
        This option will surpress mean_r, mean_g, and mean_b.
    scale : float, optional
        multiply each pixel value by the scale (this operation is performed after mean subtraction)
    random_flip : bool, optional 
        Randomly flip horizontally on the fly, useful to augment data for training neural network. 
    **kwargs :
        placeholder for new parameters

    Examples
    --------
    >>> import sframe as sf
    >>> import mxnet as mx

    >>> image_data = sf.SFrame('http://s3.amazonaws.com/dato-datasets/mnist/sframe/train')
    >>> image_data_iter = mx.io.SFrameImageIter(sframe=data, data_field=['image'], label_field='label', batch_size=100,
                                                mean_r=117, scale=0.5)

    Notes
    -----
    - Image column must contain images of the same size.
    """

    def __init__(self, sframe, data_field, label_field=None, batch_size=1,
                 data_name='data', label_name='softmax_label',
                 mean_r=0.0,
                 mean_g=0.0,
                 mean_b=0.0,
                 mean_nd=None,
                 scale=1.0,
                 random_flip=False,
                 **kwargs):
        super(SFrameImageIter, self).__init__(sframe, data_field, label_field, batch_size,
                                              data_name, label_name)

        # Mean subtraction parameters
        self._rgb_mask = np.zeros(self.data_shape)
        if mean_nd is None:
            nchannels = self.data_shape[1]
            mean_per_channel = [mean_r, mean_g, mean_b][:nchannels]
            for i in range(nchannels):
                self._rgb_mask[:, i, :, :] = mean_per_channel[i]
        elif type(mean_nd) == np.ndarray:
            mean_nd = np.swapaxes(mean_nd, 0, 2) # h, w, c -> c, w, h
            mean_nd = np.swapaxes(mean_nd, 1, 2) # c, w, h -> c, h, w
            if mean_nd.shape == self.data_shape[1:]:
                for i in range(self.data_shape[0]):
                    self._rgb_mask[i,:] = mean_nd
            else:
                raise ValueError('Shape mismatch. mean_nd has different shape from input image')
        else:
            raise TypeError('mean_nd must be type np.ndarray')
        self._rgb_mask = array(self._rgb_mask)

        # Rescale parameters
        self._scale = scale

        #Augmentation parameters
        self._random_flip = random_flip
    def _type_check(self, sframe, data_field, label_field):
        if label_field is not None:
            label_column_type = sframe[label_field].dtype()
            if label_column_type not in [int, float]:
                raise TypeError('Unexpected type for label_field \"%s\". Expect int or float, got %s' %
                                (label_field, str(label_column_type)))
        for col in data_field:
            col_type = sframe[col].dtype()
            if col_type not in [gl.Image]:
                raise TypeError('Unexpected type for data_field \"%s\". Expect or image, got %s' %
                               (col, str(col_type)))

    def _infer_column_shape(self, sarray):
        dtype = sarray.dtype()
        if not dtype is gl.Image:
            raise TypeError('Data column must be image type')

        first_image = sarray.head(1)[0]
        if first_image is None:
            raise ValueError('Column cannot contain missing value')
        return (first_image.channels, first_image.height, first_image.width)

    def iter_next(self):
        ret = super(self.__class__, self).iter_next()
        # Postprocess: normalize by mean, scale, ...
        self.data_ndarray = (self.data_ndarray - self._rgb_mask) * self._scale
        # random flip
        if self._random_flip:
            self.data_ndarray = array(self.data_ndarray.asnumpy()[:,:,:,::(np.random.randint(2)- 0.5) * 2])
        return ret


class MXDataIter(DataIter):
    """DataIter built in MXNet. List all the needed functions here.
    Parameters
    ----------
    handle : DataIterHandle
        the handle to the underlying C++ Data Iterator
    """
    def __init__(self, handle, data_name='data', label_name='softmax_label', **_):
        super(MXDataIter, self).__init__()
        self.handle = handle
        # debug option, used to test the speed with io effect eliminated
        self._debug_skip_load = False


        # load the first batch to get shape information
        self.first_batch = None
        self.first_batch = self.next()
        data = self.first_batch.data[0]
        label = self.first_batch.label[0]

        # properties
        self.provide_data = [(data_name, data.shape)]
        self.provide_label = [(label_name, label.shape)]
        self.batch_size = data.shape[0]

    def __del__(self):
        check_call(_LIB.MXDataIterFree(self.handle))

    def debug_skip_load(self):
        """Set the iterator to simply return always first batch.
        Notes
        -----
        This can be used to test the speed of network without taking
        the loading delay into account.
        """
        self._debug_skip_load = True
        logging.info('Set debug_skip_load to be true, will simply return first batch')

    def reset(self):
        self._debug_at_begin = True
        self.first_batch = None
        check_call(_LIB.MXDataIterBeforeFirst(self.handle))

    def next(self):
        if self._debug_skip_load and not self._debug_at_begin:
            return  DataBatch(data=[self.getdata()], label=[self.getlabel()], pad=self.getpad(),
                              index=self.getindex())
        if self.first_batch is not None:
            batch = self.first_batch
            self.first_batch = None
            return batch
        self._debug_at_begin = False
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        if next_res.value:
            return DataBatch(data=[self.getdata()], label=[self.getlabel()], pad=self.getpad(),
                             index=self.getindex())
        else:
            raise StopIteration

    def iter_next(self):
        if self.first_batch is not None:
            return True
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        return next_res.value

    def getdata(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetData(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl, False)

    def getlabel(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetLabel(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl, False)

    def getindex(self):
        index_size = ctypes.c_uint64(0)
        index_data = ctypes.POINTER(ctypes.c_uint64)()
        check_call(_LIB.MXDataIterGetIndex(self.handle,
                                           ctypes.byref(index_data),
                                           ctypes.byref(index_size)))
        address = ctypes.addressof(index_data.contents)
        dbuffer = (ctypes.c_uint64* index_size.value).from_address(address)
        np_index = np.frombuffer(dbuffer, dtype=np.uint64)
        return np_index.copy()

    def getpad(self):
        pad = ctypes.c_int(0)
        check_call(_LIB.MXDataIterGetPadNum(self.handle, ctypes.byref(pad)))
        return pad.value

def _make_io_iterator(handle):
    """Create an io iterator by handle."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXDataIterGetIterInfo( \
            handle, ctypes.byref(name), ctypes.byref(desc), \
            ctypes.byref(num_args), \
            ctypes.byref(arg_names), \
            ctypes.byref(arg_types), \
            ctypes.byref(arg_descs)))
    iter_name = py_str(name.value)
    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)

    doc_str = ('%s\n\n' +
               '%s\n' +
               'name : string, required.\n' +
               '    Name of the resulting data iterator.\n\n' +
               'Returns\n' +
               '-------\n' +
               'iterator: DataIter\n'+
               '    The result iterator.')
    doc_str = doc_str % (desc.value, param_str)

    def creator(*args, **kwargs):
        """Create an iterator.
        The parameters listed below can be passed in as keyword arguments.
        Parameters
        ----------
        name : string, required.
            Name of the resulting data iterator.
        Returns
        -------
        dataiter: Dataiter
            the resulting data iterator
        """
        param_keys = []
        param_vals = []

        for k, val in kwargs.items():
            param_keys.append(c_str(k))
            param_vals.append(c_str(str(val)))
        # create atomic symbol
        param_keys = c_array(ctypes.c_char_p, param_keys)
        param_vals = c_array(ctypes.c_char_p, param_vals)
        iter_handle = DataIterHandle()
        check_call(_LIB.MXDataIterCreateIter(
            handle,
            mx_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(iter_handle)))

        if len(args):
            raise TypeError('%s can only accept keyword arguments' % iter_name)

        return MXDataIter(iter_handle, **kwargs)

    creator.__name__ = iter_name
    creator.__doc__ = doc_str
    return creator

def _init_io_module():
    """List and add all the data iterators to current module."""
    plist = ctypes.POINTER(ctypes.c_void_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListDataIters(ctypes.byref(size), ctypes.byref(plist)))
    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = ctypes.c_void_p(plist[i])
        dataiter = _make_io_iterator(hdl)
        setattr(module_obj, dataiter.__name__, dataiter)

# Initialize the io in startups
_init_io_module()
