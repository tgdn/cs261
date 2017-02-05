"""

This package defines the GraphLab Create PredictiveObject, which is an abstract class
that defines the interfaces for a PredictiveObject. It also provides common
methods used by PredictiveObject, like load, save, validate etc.

"""
import abc
import os
import sys
import json
import shutil
import tempfile as _tempfie

import psclient.file_util as fu

import logging as _logging
_logger = _logging.getLogger(__name__)

try:
    from graphlab import _gl_pickle, SFrame, SArray, SGraph
except ImportError:
    try:
        from sframe import _gl_pickle, SFrame, SArray, SGraph
    except ImportError:
        raise RuntimeError("Either GraphLab-Create or SFrame package is required to create endpoint.")

from tempfile import mkdtemp

from .constants import _PREDICTIVE_OBJECT_SCHEMA_VERSION

class PredictiveObject(object):
    '''Predictive Object Abstraction

    Derived class needs to implement the following interface:
      * query() -- given input, return query result
      * get_doc_string() -- returns documentation for the Predictive Object
    '''
    __metaclass__ = abc.ABCMeta

    _depedent_file_path = None

    def __init__(self, description = ''):
        self.required_packages = []
        self.required_files = {}
        self.description = description
        self.schema_version = _PREDICTIVE_OBJECT_SCHEMA_VERSION

    def __repr__(self):
        """Returns a string description of the object."""
        ret = ""
        ret = ret + 'class             :' + self._get_full_class_name() + "\n"
        ret = ret + 'description       :' + self.description + "\n"
        ret = ret + 'required packages :' + str(self.required_packages) + "\n"
        return ret

    def __str__(self):
        """Returns a string description of the object."""
        return self.__repr__()

    def get_dependencies(self):
        '''All endpoints this endpoint depends on'''
        return []

    def get_ml_metric_config(self):
        '''Get the configuration for all the ml_metrics being tracked.'''
        return {}

    @abc.abstractmethod
    def query(self, input):
        """Given input, get the prediction for the input"""
        pass

    @abc.abstractmethod
    def get_doc_string(self):
        '''Returns documentation for the predictive object query

        By default, this method returns the docstring for 'query' method
        Derived class may overwrite this method to dynamically create doc string
        '''
        pass

    def save(self, path, aws_credentials = {}):
        """ Save predictive object to the given path

        Parameters
        ----------
        path : str
          The location to save the predictive object to
        """
        # only support saving to local or S3 for now
        if (not (fu.is_s3_path(path) or \
                 fu.is_local_path(path) or \
                 fu.is_hdfs_path(path))):
            raise RuntimeError("Only save to local, hdfs and S3 path are supported, "
                        "cannot save predictive object to path %s. " % path)

        if fu.is_local_path(path) and os.path.exists(path):
            if os.path.exists(path):
                _logger.warning("Overwriting existing file '%s' when saving predictive object" % path)
                rm_fn = os.remove if os.path.isfile(path) else shutil.rmtree
                rm_fn(path)

        if fu.is_local_path(path):
            self._save_local(path)
        else:
            self._save_remote(path, aws_credentials)

    def _save_remote(self, path, aws_credentials):
        '''Save current predictive object to S3
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
        '''Save current predictive object to local path
        '''
        pickler = _gl_pickle.GLPickler(path)
        schema_version = self.schema_version
        required_files = self.required_files

        try:
            pickler.dump(schema_version)
            pickler.dump(required_files)
            pickler.dump(self)
        except Exception as e:
            _logger.error('Unable to save object: %s' % (e.message))
            raise e
        finally:
            pickler.close()

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
            unpickler = _gl_pickle.GLUnpickler(path)
            po_schema_version = unpickler.load()
            required_files = unpickler.load()

            # layout the required files before loading the function
            # Required files are moved to be managed separatly from the Predictive
            # object since schema version 4 (jumped to 6 to be inline with PS version)
            # so no longer need to deserialize the required files as part of load
            if po_schema_version <= 3:
                cls._deserialize_required_files(required_files)
            else:
                # Do not need to load the dependent files after schema version 3
                # because the loading is handled in higher level by Predictive
                # service
                pass
            po_obj = unpickler.load()
            unpickler.close()

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            err_msg = "Traceback\n %s\n" % trace
            err_msg += "Error type    : %s\n" % e.__class__.__name__
            err_msg += "Error message : %s\n" % str(e)

            raise RuntimeError('Unable to load predictive object. Error: %s' % err_msg)

        if po_schema_version > _PREDICTIVE_OBJECT_SCHEMA_VERSION:
            raise RuntimeError("Your GraphLab Create only supports Predictive "
                "Objects with schema version up to '%s', the Predictive Object "
                "you are trying to load has schema version '%s'. Please upgrade your "
                "GraphLab Create version to most up-to-date one." % \
                (_PREDICTIVE_OBJECT_SCHEMA_VERSION, po_schema_version))

        return po_obj

    @classmethod
    def _deserialize_required_files(cls, required_files):
        '''
        Desrialize required files.
        This is only used for Predictive Object upto version 1.6. Starting from
        1.7, the required_files are shipped by Predictive Service on behalf of
        the Predictive Object so the Predictive Object does not need to contain
        the files anymore.

        For backward compatibility, we need to be able to keep this functionality
        so that we can read in the dependent files
        '''
        if len(required_files) == 0:
            _logger.debug("no additional required files.")
            return None

        dependent_file_path = cls._get_dependent_file_path()
        _logger.info('_deserialize_required_files to %s: %s' % (dependent_file_path, required_files.keys()))

        cwd = os.getcwd()
        try:
            # This is required so that the sys.module knows to load the module from
            # current directory
            os.chdir(dependent_file_path)

            reload_modules = []
            for filename in required_files.keys():
                f_name = os.path.join(dependent_file_path, filename)

                if not os.path.exists(os.path.dirname(f_name)):
                    os.makedirs(os.path.dirname(f_name))

                _logger.info('deserializing %s' % f_name)
                with open(f_name, 'w') as f:
                    f.write(required_files[filename])

                fileName, fileExtension = os.path.splitext(filename)
                if fileExtension == '.py' or fileExtension == '.pyc':
                    module_name = fileName.replace('/', '.')
                    reload_modules.append(module_name)

            # reload all modules, need to do after all files are written, otherwise
            # some files may not be able to be found
            _logger.info('reload dependent modules')
            for module_name in reload_modules:
                try:
                    m = sys.modules.get(module_name)
                    if m:
                        _logger.info("removing module: %s" % module_name)
                        del sys.modules[module_name]

                    _logger.info("Importing module: %s" % module_name)
                    reload(__import__(module_name))
                except Exception as e:
                    _logger.info(e)
                    _logger.info("ignore reloading of module %s" % module_name)
                    pass

        except Exception as e:
            _logger.error("Failed to deserialize required files, error: %s" % e.message)
            raise
        finally:
            os.chdir(cwd)

    @classmethod
    def _get_dependent_file_path(cls):
        '''We put all dependent_files under a fixed path so that newer file may
        overwrite older files to guarantee the modules can be reloaded correctly
        '''

        if PredictiveObject._depedent_file_path is None:
            PredictiveObject._depedent_file_path = mkdtemp(prefix='predictive_object_files')
            sys.path.insert(1, PredictiveObject._depedent_file_path)

        return PredictiveObject._depedent_file_path

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

    def _get_full_class_name(self):
        module = self.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return self.__class__.__name__
        return module + '.' + self.__class__.__name__

    @classmethod
    def _make_serializable(cls, result):
        '''Convert a result from model query to python data structure that can
        easily serialize over network
        '''
        ret = result
        if type(result) == SFrame:
            # SFrame is converted to list of dictionary
            if len(result.column_names()) > 1:
                ret = list(result.pack_columns(dtype=dict)['X1'])
            elif result.num_rows() == 0:
                ret = []
            else:
                col = result.column_names()[0]
                ret = [{col: i} for i in result[col]]
        elif type(result) == SArray:
            # SArray is converted to list directly
            ret = list(result)
        elif type(result) == SGraph:
            raise RuntimeError("SGraph is not supported as query result")

        try:
            json.dumps(ret)
        except:
            raise TypeError("Result from model query is not json serializable: %s" % ret)

        return ret

    # Returns an array of dictionary that contains the methods and their corresponding
    # schema information.
    @abc.abstractmethod
    def get_methods(self):
        return None
