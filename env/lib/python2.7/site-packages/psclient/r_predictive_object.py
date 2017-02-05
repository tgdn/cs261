import os
import shutil
import logging as _logging
import tempfile as _tempfie
_logger = _logging.getLogger(__name__)

import psclient.file_util as fu
from .constants import _PREDICTIVE_OBJECT_SCHEMA_VERSION

class RPredictiveObject(object):

    def __init__(self, po_path, description = ''):
        self.po_path = po_path
        self.schema_version = _PREDICTIVE_OBJECT_SCHEMA_VERSION
        self.description = description

    def __repr__(self):
        """Returns a string description of the object."""
        ret = ""
        ret = ret + 'class             :' + self._get_full_class_name() + "\n"
        ret = ret + 'description       :' + self.description + "\n"
        return ret

    def __str__(self):
        """Returns a string description of the object."""
        return self.__repr__()

    def get_dependencies(self):
        '''All endpoints this endpoint depends on'''
        return []

    def _get_full_class_name(self):
        module = self.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return self.__class__.__name__
        return module + '.' + self.__class__.__name__

    def get_methods(self):
        return [{'method': 'query','schema': {'sample': None, 'input': None, 'output': None}}]

    def get_doc_string(self):
        return " "

    def save(self, path, aws_credentials = {}):
        """ save R predictive object to the given path

        parameters
        ----------
        path : str
          the location to save the predictive object to
        """
        # only support saving to local or s3 for now
        if (not (fu.is_s3_path(path) or \
                 fu.is_local_path(path) or \
                 fu.is_hdfs_path(path))):
            raise RuntimeError("only save to local and s3 path is supported, cannot "
              "save predictive object to path %s. " % path)

        if fu.is_local_path(path) and os.path.exists(path):
            if os.path.exists(path):
                _logger.warning("overwriting existing file '%s' when saving predictive object" % path)
                rm_fn = os.remove if os.path.isfile(path) else shutil.rmtree
                rm_fn(path)

        if fu.is_local_path(path):
            self._save_local(path)
        else:
            self._save_remote(path, aws_credentials)

    def _save_remote(self, path, aws_credentials):
        '''save current predictive object to s3
        '''
        tempdir = _tempfie.mkdtemp(prefix='predictive_object_')
        try:
            self._save_local(tempdir)
            if fu.is_s3_path(path):
                fu.upload_to_s3(tempdir, path, is_dir=True, aws_credentials = aws_credentials)
            elif fu.is_hdfs_path(path):
                fu.hdfs_mkdir(path)
                fu.upload_to_hdfs(tempdir + '/*', path)
        finally:
            shutil.rmtree(tempdir)

    def _save_local(self, path):
        '''save R predictive object path to local path
        '''
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            # Write the version number.
            with open(os.path.join(path, 'version'), 'w') as f:
                f.write("1.0")
            shutil.copy(self.po_path, os.path.join(path, "pickle_archive"))
        except Exception as e:
            _logger.error('unable to save serialized predictive object %s into %s' % (self.po_path, path))
            raise e

    @classmethod
    def _load_remote(cls, path, schema_version, aws_credentials={}):
        temp_dir = _tempfie.mkdtemp(prefix='predictive_object_')

        if fu.is_s3_path(path):
            fu.download_from_s3(path, temp_dir, is_dir=(schema_version > 2),
                              aws_credentials=aws_credentials)
        elif fu.is_hdfs_path(path):
            fu.download_from_hdfs(path, temp_dir, is_dir=(schema_version > 2))
        else:
            assert 'Only support S3 and HDFS path for Predictive Object saving location!'

        return cls._load_local(temp_dir)

    @classmethod
    def load(cls, path, schema_version, aws_credentials={}):
        """ Load predictive object from given path
        """
        new_po = None
        if (fu.is_local_path(path)):
            new_po = cls._load_local(path)
        else:
            new_po = cls._load_remote(path, schema_version, aws_credentials)

        _logger.info('Loaded predictive object "%s" successfully' % type(new_po).__name__)

        return new_po

    @classmethod
    def _load_local(cls, path):
        path = fu.expand_full_path(path)
        if not os.path.exists(path):
            raise RuntimeError("Path %s does not exist." % path)

        try:
            po_obj = RPredictiveObject(os.path.join(path, "pickle_archive"))

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            err_msg = "Traceback\n %s\n" % trace
            err_msg += "Error type    : %s\n" % e.__class__.__name__
            err_msg += "Error message : %s\n" % str(e)

            raise RuntimeError('Unable to load predictive object. Error: %s' % err_msg)

        return po_obj
