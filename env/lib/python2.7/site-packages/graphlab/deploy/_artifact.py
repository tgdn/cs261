import graphlab
from graphlab import _gl_pickle as gl_pickle

def load_artifact(filename):
    """
    Load an artifact object.

    Parameters
    ----------
    filename : A GLPickle archive filename.

    """
    unpickler = gl_pickle.GLUnpickler(filename)

    # Get the version
    version = unpickler.load()

    # Load the constructor that knows how to load a empty object
    post_import_cls = unpickler.load()

    # Construct an empty object by following the module to the constructor
    module = eval(post_import_cls.__module__)   # requires graphlab has been imported
    cls = module.__getattribute__(post_import_cls.__name__)

    # Load the object with the right version.
    obj = cls._load_version(unpickler, version)

    unpickler.close()

    # Return the object
    return obj

class Artifact(object):
    """
    Any object that needs to be managed by a GraphLab session.

    Sessions provide functionality for managing the lifecycle of:
    (1) Tasks
    (2) Environments
    (3) Jobs

    and in the future, any other new objects.
    """

    #: name
    name = None

    # All objects must know if they have been modified or not.
    _modified_since_last_saved = None

    # Back pointers to the session object
    _session = None

    # Created date
    _session_registration_date = None

    # Used by the scoped session to seggregate various object types. For
    # example: LocalAsync, Local, EC2, and Hadoop all must have typename
    # Environment even though they are of different types.
    _typename = 'Artifact'

    def __init__(self):
        pass
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return 'Artifact(%s, %s)' % (self.name, self._typename)

    def __getstate__(self):
        """
        Object serialization routines. Do not pickle the session with the object.
        Bad things happen.
        """
        odict = dict.copy(self.__dict__)
        if '_session' in odict:
            del odict['_session']
        return odict

    def _save_to_file(self, filename):
        """
        Save (version controlled) the object.

        Parameters
        ----------
        filename : Filename to save (gl_pickle archive file type)

        Notes
        ----------
        All classes are automatically versioned.
        """
        # Setup the GLC pickler
        pickler = gl_pickle.GLPickler(filename)

        pickler.dump(self._get_version())
        pickler.dump(self.__class__)

        # Each child class must implement save_version
        self._save_impl(pickler)
        pickler.close()

    def _save_impl(self, pickler):
        """
        An abstract function to implement save for the object in consideration.
        The default implementation will dump self to the pickler.

        Overwrite for smarter implementations.

        Parameters
        ----------
        pickler : An opened gl_pickle archive (DO NOT CLOSE after completion)
        """
        pickler.dump(self)

    def _get_version(self):
        raise NotImplementedError

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : A GLUnpickler file handle.

        version : Version number as maintained by the class.

        """
        raise NotImplementedError

    def _get_metadata(self):
        """
        Get the metadata that is managed by the session. This gets displayed
        in the Scoped session and is stored in the session's index file.

        """
        status = 'Yes' if self._modified_since_last_saved == True else 'No'
        return {'Name': self.name,
                'Type': type(self).__name__,
                'Unsaved changes?' : status,
                'Creation date': self._session_registration_date}

    def __eq__(self, obj):
        return obj.__class__ == self.__class__ and obj.name == self.name

    def _set_dirty_bit(self):
        self._modified_since_last_saved = True
