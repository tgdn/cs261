import os as _os
import logging as _logging
import json as _json
from collections import namedtuple as _namedtuple
from .model import FeedForward as _FeedForward
from .model import extract_feature as _extract_feature
from . import ndarray as _ndarray
from . import io as _io
from . import symbol as _sym
from .utils.rcnn_utils import nms, bbox_transform_inv, clip_boxes
from .utils import proposal
import numpy as _np
import requests
import collections

_LOGGER = _logging.getLogger(__name__)

class ModelTypeEnum(object):
    """
    Enumeration of Pretrained Model Type
    """
    IMAGE_CLASSIFIER= 'IMAGE_CLASSIFIER'
    IMAGE_DETECTOR = 'IMAGE_DETECTOR'

ModelEntry = _namedtuple('ModelEntry', ['name', 'type', 'version'])


"""
Default location for model download
"""
DEFAULT_MODEL_LOCATION=_os.path.expanduser('~/.graphlab/mxnet_models')


def list_models(location=DEFAULT_MODEL_LOCATION):
    """
    Return list of pretrained model names.

    Parameters
    ----------
    location : str, optional
      The local directory where the model is saved to.

    Examples
    --------
    >>> mx.pretrained_model.list_models()
    """
    if not _os.path.exists(location):
        _os.makedirs(location)

    models = [p for p in _os.listdir(location)]
    ret = []
    for name in models:
        model_path = _os.path.join(location, name)
        if not _os.path.isdir(model_path):
            continue
        metadata_path = _os.path.join(model_path, 'metadata.json')
        version = None
        try:
            f = open(metadata_path)
            metadata = _json.load(f)
            version = metadata['version']
            model_type = metadata['model_type']
            name = metadata['name']
            ret.append(ModelEntry(name, model_type, version))
        except Exception as e:
            _LOGGER.warn('Unable to open or parse model metadata %s.' % metadata_path)
    return ret

def _download_file(url, target_file):
    r = requests.get(url, stream=True)
    with open(target_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def download_model(url, location=DEFAULT_MODEL_LOCATION, overwrite=False):
    """
    Perform downloading the model to local filesystem.

    Parameters
    ----------
    url : str,
      URL of the model. Get the list of available models from
      https://dato.com/products/create/docs/graphlab.mxnet.pretrained_image_model.html
    location : str, optional
      The local directory where the model is saved to.
    overwrite : bool, optional
      If true, remove existing models first.

    Examples
    --------
    >>> mx.pretrained_model.download_model('https://.../image_classifier/imagenet1k_inception_v3-1.0.tar.gz')
    >>> model = mx.pretrained_model.load_model('imagenet1k_inception_v3')
    """
    name = url.split('/')[-1]
    name = name.split('-')[0]
    target_dir = _os.path.join(location, name)
    target_exists = _os.path.exists(target_dir)
    if overwrite is False and target_exists:
        raise OSError("Target directory %s already exists. Please remove existing model or set overwrite to True" % target_dir)
    # Peform download
    _LOGGER.info("Begin downloading %s" % (url))
    import tempfile
    import hashlib
    import shutil
    import tarfile
    import sys

    f = tempfile.NamedTemporaryFile(suffix='tar.gz')
    _download_file(url, f.name)
    _LOGGER.info("Filed downloaded to %s" % (f.name))

    # Extract
    if target_exists:
        _LOGGER.info("Remove existing model: %s" % target_dir)
        shutil.rmtree(target_dir)
    _os.makedirs(target_dir)
    _LOGGER.info("Extracting model to %s" % target_dir)
    tar = tarfile.open(f.name)
    tar.extractall(location)
    tar.close()
    _LOGGER.info("Model %s is downloaded to %s" % (name, target_dir))


def load_model(name, ctx=None, location=DEFAULT_MODEL_LOCATION):
    """
    Load a pretrained model by name.

    Parameters
    ----------
    name : str
        Name of the pretrained model. Model must be downloaded first
    ctx : mx.context, optional
        Context of the model. Default None is equivalent to mx.cpu()
    location : str, optional
        The directory which contains downloaded models

    Examples
    --------
    >>> model = mx.pretrained_model.load_model('imagenet1k_inception_v3', ctx=mx.gpu(0))
    """
    if not any(name == m.name for m in list_models(location)):
        raise KeyError('Model %s does not exist. Models can be listed using list_models()' % name)
    target_dir = _os.path.join(location, name)
    return load_path(target_dir, ctx)


def load_path(target_dir, ctx=None):
    """
    Load a pretrained model by path.

    Parameters
    ----------
    path : str
        Path of the downloaded pretrained model.
    ctx : mx.context, optional
        Context of the model. Default None is equivalent to mx.cpu().

    Examples
    --------
    >>> model = mx.pretrained_model.load_path('~/.graphlab/mxnet_models/imagenet1k_inception_v3')
    """

    _LOGGER.debug('load from: %s' % (target_dir))
    target_dir = _os.path.expanduser(target_dir)

    # Load the model metadata
    metadata_path = _os.path.join(target_dir, 'metadata.json')
    _LOGGER.debug('metadata_path: %s' % metadata_path)
    metadata = _json.load(open(metadata_path))
    _LOGGER.debug('metadata: %s' % str(metadata))
    if metadata['mean_nd'] is not None:
        mean_nd_path = _os.path.join(target_dir, metadata['mean_nd'])
        mean_nd = _ndarray.load(mean_nd_path)['mean_img'].asnumpy()
        # mean_nd is in c, h, w order. need to convert to h, w, c
        mean_nd = _np.swapaxes(mean_nd, 0, 2) # c, h, w -> w, h, c
        mean_nd = _np.swapaxes(mean_nd, 0, 1) # w, h, c -> h, w, c
        metadata['mean_nd'] = mean_nd

    # Load Image Classifier
    if metadata['model_type'] == ModelTypeEnum.IMAGE_CLASSIFIER:
        param_file = [f for f in _os.listdir(target_dir) if f.endswith('.params')]
        if len(param_file) != 1:
            raise ValueError('Invalid model directory %s. Please remove the directory and redownload the model' % target_dir)

        # Parse the file name to get prefix and epoch
        _LOGGER.debug('param_file: %s' % param_file[0])
        prefix = _os.path.splitext(param_file[0])[0]
        epoch = prefix.split('-')[-1]
        prefix = prefix[:-(len(epoch) + 1)]
        prefix = _os.path.join(target_dir, prefix)
        epoch = int(epoch)
        _LOGGER.debug('prefix: %s, epoch: %s' % (prefix, epoch))

        # Load the feedforward model
        model = _FeedForward.load(prefix, epoch, ctx)

        # Load the labels
        label_file = _os.path.join(target_dir, 'labels.json')
        _LOGGER.debug('label_file: %s' % label_file)
        labels = _json.load(open(label_file))
        return ImageClassifier(model, labels, metadata)
    elif metadata['model_type'] == ModelTypeEnum.IMAGE_DETECTOR:
        param_file = [f for f in _os.listdir(target_dir) if f.endswith('.params')]
        if len(param_file) != 1:
            raise ValueError('Invalid model directory %s. Please remove the directory and redownload the model' % target_dir)
        _LOGGER.debug('param_file: %s' % param_file[0])
        prefix = _os.path.splitext(param_file[0])[0]
        epoch = prefix.split('-')[-1]
        prefix = prefix[:-(len(epoch) + 1)]
        prefix = _os.path.join(target_dir, prefix)
        epoch = int(epoch)
        _LOGGER.debug('prefix: %s, epoch: %s' % (prefix, epoch))

        # Load the feedforward model
        model = _FeedForward.load(prefix, epoch, ctx)
        label_file = _os.path.join(target_dir, 'labels.json')
        _LOGGER.debug('label_file: %s' % label_file)
        labels = _json.load(open(label_file))
        return ImageDetector(model, labels, metadata)
    else:
        raise TypeError('Unexpected model type: %s', metadata['model_type'])


class ImageClassifier(object):
    """
    Wrapper of pretrained image classifier model.

    Use :py:func:`load_model` or :py:func:`load_path` to load the model. Do not
    construct directly.

    Parameters
    ----------
    model :  FeedForward
        The underlying model
    labels : list
        Map from index to label
    metadata : dict
        Metadata of the model including name, version, input shape, etc.
    """
    def __init__(self, model, labels, metadata):
        self._model = model
        self._labels = labels
        self._input_shape = metadata['input_shape']
        self.mean_nd = metadata['mean_nd']
        self.mean_rgb = metadata['mean_rgb']
        self.rescale = metadata['scale']
        self._name = metadata['name']
        self._version = metadata['version']
        if 'label_name' in metadata:
            self._label_name = metadata['label_name']
        else:
            self._label_name = 'softmax_label'
        if 'data_name' in metadata:
            self._data_name = metadata['data_name']
        else:
            self._data_name = 'data'

    @property
    def model(self):
        return self._model

    @property
    def labels(self):
        return self._labels

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def __str__(self):
        return "ImageClassifier: " + self._name + "(version %s)" % self._version

    def __repr__(self):
        return self.__str__()

    def _make_dataiter(self, data, batch_size):
        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Require GraphLab Create or SFrame')

        if type(data) is not _gl.SFrame and type(data) is not _gl.SArray and type(data) is not _gl.Image:
            raise TypeError('Input data must be SFrame, SArray or Image')
        if type(data) is _gl.SArray and data.dtype() != _gl.Image:
            raise TypeError('Expect image typed SArray, actual type is %s' % str(data.dtype()))
        elif type(data) is _gl.SFrame:
            if data.column_types().count(_gl.Image) != 1:
                raise TypeError('Input SFrame must contain a single Image typed column')
        if type(data) is _gl.SArray:
            data = _gl.SFrame({'image': data})
        if type(data) is _gl.Image:
            data = _gl.SFrame({'image': [data]})
        if batch_size > len(data):
            batch_size = len(data)

        image_col = data.column_names()[data.column_types().index(_gl.Image)]
        first_image = data[image_col][0]
        input_shape = (first_image.height, first_image.width, first_image.channels)

        if input_shape != tuple(self.input_shape):
            _LOGGER.info('Detect image shape mismatches network input shape. Perform resize to shape %s' \
                    % str(tuple(self.input_shape)))
            data_resize = _gl.SFrame()
            data_resize[image_col] = _gl.image_analysis.resize(data[image_col],
                                                               self.input_shape[0],
                                                               self.input_shape[1],
                                                               self.input_shape[2], decode=True)
            data = data_resize

        dataiter = _io.SFrameImageIter(data, data_field=[image_col],
                                       batch_size=batch_size,
                                       mean_r=self.mean_rgb[0],
                                       mean_g=self.mean_rgb[1],
                                       mean_b=self.mean_rgb[2],
                                       mean_nd=self.mean_nd,
                                       data_name=self._data_name,
                                       label_name=self._label_name,
                                       scale=self.rescale)
        return dataiter

    def extract_feature(self, *args, **kwargs):
        """
        Alias for :py:func:`~mxnet.pretrained_model.ImageClassifier.extract_features`
        """
        return self.extract_features(*args, **kwargs)

    def extract_features(self, data, batch_size=50):
        """
        Extracts features from the second to last layer in the network.

        Parameters
        ----------
        data : SFrame, SArray[Image] or Image
            SFrame with a single image typed column, an SArray of Images, or
            a single Image. The images can be of various sizes.
       batch_size : int, optional
            batch size of the input to the internal model. Larger
            batch size makes the prediction faster but requires more memory.

        Returns
        -------
        out : SArray
            An SArray of feature

        Examples
        --------
        >>> m = mx.pretrained_model.load_model('mnist_lenet')
        >>> sf = SFrame('http://s3.amazonaws.com/dato-datasets/mnist/sframe/train')
        >>> m.extract_feature(sf['image'])
        """
        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Require GraphLab Create or SFrame')

       # Make DataIter
        dataiter = self._make_dataiter(data, batch_size)
        features =  _extract_feature(self.model, dataiter)

        ret = _gl.SArray(features)
        return ret

    def predict_topk(self, data, k=5, batch_size=50):
        """
        Predict the topk classes for given data

        Parameters
        ----------
        data : SFrame or SArray[Image]
            SFrame with a single image typed column, an SArray of Images.
            or a single Image. The images can be of various sizes.
        k : int, optional
            Number of classes returned for each input
        batch_size : int, optional
            batch size of the input to the internal model. Larger
            batch size makes the prediction faster but requires more memory.

        Returns
        -------
        out : SFrame
            An SFrame of 5 columns: row_id, label_id, label, score, rank

        Examples
        --------
        >>> m = mx.pretrained_model.load_model('mnist_lenet')
        >>> sf = SFrame('http://s3.amazonaws.com/dato-datasets/mnist/sframe/train')
        >>> m.predict_topk(sf['image'])
        """
        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Require GraphLab Create or SFrame')

        # Check input
        if k > len(self.labels):
            k = len(self.labels)

        # Make DataIter
        dataiter = self._make_dataiter(data, batch_size)

        # Make prediction
        # pred[i][j] is the score of row i belongs to label j
        pred = self.model.predict(dataiter)
        # top_idx[i][k] is the label index of kth highest score of row i
        top_idx = pred.argsort()[:,-k:][:,::-1]
        # Take row wise index, to get topk score for each row
        top_scores = pred[_np.arange(pred.shape[0])[:,None], top_idx]

        top_labels = [self.labels[i] for i in top_idx.flatten()]
        row_ids = _np.repeat(_np.arange(len(pred)), k)
        ranks = _np.tile(_np.arange(1, k+1), len(pred))

        ret = _gl.SFrame()
        ret['row_id'] = row_ids
        ret['label'] = top_labels
        ret['score'] = top_scores.flatten()
        ret['rank'] = ranks
        return ret

class ImageDetector(object):
    """
    Wrapper of pretrained image detect model.

    Use :py:func:`load_model` or :py:func:`load_path` to load the model. Do not
    construct directly.

    Parameters
    ----------
    model: FeedForward
        The underlying model
    labels : list
        Map from index to label
    metadata: dict
        Metadata of the model including name, version, input shape, etc
    """
    def __init__(self, model, labels, metadata):
        self._model = model
        self._labels = labels
        self._metadata = metadata
        self._num_classes = len(labels)
        # rcnn params
        for k, v in metadata.items():
            self.__setattr__("_" + k, v)
        if "target_size" not in metadata:
            self._target_size = 600
        if "max_size" not in metadata:
            self._max_size = 1000

        # new executor
        self._ctx = model.ctx[0] # only support single device for now
        self._sym = model.symbol
        self._base_sym = model.symbol
        self._arg_params = {k : v.as_in_context(self._ctx) for k, v in model.arg_params.items()}
        self._aux_params = {k : v.as_in_context(self._ctx) for k, v in model.aux_params.items()}
        self._executor_with_feature = False

    def _init_executor(self, sym):
        self._executor = sym.bind(self._ctx, self._arg_params, args_grad=None,
                                  grad_req="null", aux_states=self._aux_params)

    @property
    def model(self):
        return self._model

    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def __str__(self):
        return "ImageDetector: " + self._name + "(version %s)" % self._version

    def _detect(self, gl_im):
        im_tensor, im_info = self._preprocess(gl_im)
        self._arg_params["im_info"][:] = im_info
        self._arg_params["data"] = _ndarray.array(im_tensor, ctx=self._ctx)
        self._init_executor(self._sym)
        self._executor.forward(is_train=False)
        blobs = dict(zip(self._sym.list_outputs(), [x.asnumpy() for x in self._executor.outputs]))
        scores = blobs["cls_prob_output"]
        bbox_deltas = blobs["bbox_pred_output"]
        rois = blobs["rois_output"]
        boxes = rois[:, 1:5] / im_info[0][2]
        boxes = bbox_transform_inv(boxes, bbox_deltas)
        boxes = clip_boxes(boxes, (im_info[0][0] / im_info[0][2],
                                   im_info[0][1] / im_info[0][2],
                                   3))
        blobs["bbox_pred"] = boxes
        del blobs["rois_output"]
        del blobs["bbox_pred_output"]
        return blobs


    def _preprocess(self, gl_im):
        import PIL
        import numpy as np
        # build PIL image from sframe
        im = gl_im._to_pil_image()
        im_shape = im.size
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self._target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > self._max_size:
            im_scale = float(self._max_size) / float(im_size_max)
        im_hw = (int(im_shape[0] * im_scale), int(im_shape[1] * im_scale))
        im = im.resize(im_hw, PIL.Image.ANTIALIAS)
        # normalize
        im = np.asarray(im).astype("float32").copy()
        im -= self._mean_rgb
        if "scale" in self._metadata:
            im *= self._scale
        im_tensor = im[np.newaxis, :]
        im_tensor = im_tensor.transpose((0, 3, 1, 2))
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        return im_tensor, im_info


    def _postprocess(self, blobs, class_score_threshold=0.5, nms_threshold=0.3):
        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Require GraphLab Create or SFrame')

        out_blob = {"class":[], "box":[], "score":[], "feature":[]}
        scores = blobs["cls_prob_output"]
        boxes = blobs["bbox_pred"]
        for j in range(1, self._num_classes):
            inds = _np.where(scores[:, j] > class_score_threshold)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4 : (j + 1) * 4]
            cls_dets = _np.hstack((cls_boxes, cls_scores[:, _np.newaxis]))\
                    .astype(_np.float32, copy=False)
            keep = nms(cls_dets, nms_threshold)

            # make result
            out_blob["class"].extend([self._labels[j] for i in range(len(keep))])
            out_blob["box"].extend(list(cls_boxes[keep, :]))
            out_blob["score"].extend(cls_scores[keep])
            if self._executor_with_feature:
                fea =  blobs["feature_output"][keep, :]
                out_blob["feature"].extend(list(fea))
        if not self._executor_with_feature:
            del out_blob["feature"]
        return _gl.SFrame(out_blob)

    def _detect_array(self, arr, class_score_threshold=0.5, nms_threshold=0.3):
        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Require GraphLab Create or SFrame')
        ret = _gl.SFrame()
        det_cnt = [0] * len(arr)
        for i in range(len(arr)):
            tmp = self._postprocess(self._detect(arr[i]), class_score_threshold, nms_threshold)
            det_cnt[i] = len(tmp)
            # if nothing detected:
            if len(tmp) == 0:
                if self._executor_with_feature:
                    tmp = _gl.SFrame({"box":[[0.,0.,0.,0.]],
                        "class":["nothing"], "score":[0.], "feature":[[0.]]})
                else:
                    tmp = _gl.SFrame({"box":[[0.,0.,0.,0.]], "class":["nothing"], "score":[0.]})
                det_cnt[i] = 1
            ret = ret.append(tmp)
        image_id = []
        for i in range(len(arr)):
            image_id.extend([i] * det_cnt[i])
        ret.add_column(_gl.SArray(image_id), "id")
        return ret


    def detect(self, data, class_score_threshold=0.5, nms_threshold=0.3):
        """
        Detect objects for the given data

        Parameters
        ----------
        data : SFrame, SArray[Image] or gl.Image
            SFrame, SArray of images type or a single gl.Image
            Image can be of various sizes.
        class_score_threshold: float, optional
            Threshold for filtering.
            If the classification score is below threshold, the result will be filtered out
        nms_threshold: float, optional
            Threshold for filtering by nms(non-maximum suppression)
            If the nms score is below threshold, the result will be filtered out

        Returns
        -------
        out: SFrame
            SFrame with column: id, class, box, score, feature
        """

        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Require GraphLab Create or SFrame')
        if type(data) == _gl.Image:
            return self._detect_array([data], class_score_threshold, nms_threshold)
        elif type(data) == _gl.SArray:
            return self._detect_array(data, class_score_threshold, nms_threshold)
        elif type(data) == _gl.SFrame:
            if data.column_types().count(_gl.Image) != 1:
                raise TypeError('Input SFrame must contain a single Image typed column')
            sa_data = data[data.column_names()[data.column_types().index(_gl.Image)]]
            return self._detect_array(sa_data, class_score_threshold, nms_threshold)
        else:
            raise Exception("Unsupported input data type.")

    def extract_feature(self, *args, **kwargs):
        """
        Alias for :py:func:`~mxnet.pretrained_model.ImageDetector.extract_features`
        """
        return self.extract_features(*args, **kwargs)


    def extract_features(self, data,
                        class_score_threshold=0.5,
                        nms_threshold=0.3,
                        layer="roi_pool5"):
        """
        Detect objects and extract feature of objects for the given data

        Parameters
        ----------
        data : SFrame, SArray[Image] or gl.Image
            SFrame, SArray of images type of a single gl.Image
            Image can be of various sizes.
        class_score_threshold: float, optional
            Threshold for filtering.
            If the classification score is below threshold, the result will be filtered out
        nms_threshold: float, optional
            Threshold for filtering by nms(non-maximum suppression)
            If the nms score is below threshold, the result will be filtered out
        layer: str, optional
            Name of layer which generates object feature


        Returns
        -------
        out: SFrame
            SFrame with column: id, class, box, score, feature
        """

        if not self._executor_with_feature:
            outputs = self._sym.list_outputs()
            internals = self._sym.get_internals()
            roi_pool = internals["%s_output" % layer]
            fea_pool = _sym.Pooling(name="roi_avg_pool", data=roi_pool,
                                      kernel=(7,7), stride=(1,1), pool_type="avg")
            fea_flatten = _sym.Flatten(name="feature", data=fea_pool)
            tmp = [internals[x] for x in outputs]
            tmp.append(fea_flatten)
            self._sym = _sym.Group(tmp)
            self._executor_with_feature = True
        ret = self.detect(data, class_score_threshold, nms_threshold)
        self._sym = self._base_sym
        self._executor_with_feature = False
        return ret

    def visualize_detection(self, gl_im, dets):
        """
        Visualize detection result

        Parameters
        ----------
        gl_im: gl.Image
            The image to be visualized
        dets: SFrame
            detection result in sframe

        Returns
        -------
        out: gl.Image
            Image with bounding boxes drawn.

        """
        try:
            from PIL import ImageDraw,ImageFont
        except ImportError:
            raise ImportError('PIL or pillow package is required for this')

        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Requires GraphLab Create or SFrame')

        _format = {'JPG': 0, 'PNG': 1, 'RAW': 2, 'UNDEFINED': 3}


        assert(type(gl_im) == _gl.Image)
        pil_img = gl_im._to_pil_image()
        draw = ImageDraw.Draw(pil_img)
        fnt = ImageFont.truetype(_os.path.join(_os.path.dirname(__file__),'Lato-Regular.ttf'), 12)
        assert len(dets['id'].unique()) == 1, "only support visualize single image"

        for row in dets:
            bbox = row['box']
            cls = str(row['class'])
            score = str(row['score'])

            draw.rectangle(list(bbox),outline="red")
            draw.rectangle([bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1], outline="red")
            draw.rectangle([bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2], outline="red")
            draw.text((bbox[0] + 2, bbox[1] - 16),cls + ":" + score, font=fnt, fill="red")

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

        # Construct a graphlab.Image

        return _gl.Image(_image_data=image_data, _width=width, _height=height, _channels=channels, _format_enum=format_enum, _image_data_size=image_data_size)


