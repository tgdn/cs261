import logging as _logging
import sys as _sys
import graphlab as _gl
from . import _internal_utils
from ._artifact import Artifact as _Artifact
from graphlab.util import _raise_error_if_not_of_type

# We keep track of version.
# -----------------------------------------------------------------------------
#
#  Version 1: GLC 1.3
#  --------------------
#  Everything starts from scratch here. Everything before V1.3 is not version
#  controlled and hence, will not be fowards compatible.
#
#
__LOGGER__ = _logging.getLogger(__name__)

class _Environment(_Artifact):
    """Base class for environments for running Tasks / Jobs."""

    # All environments must have these parameters.

    _typename = 'Environment'
    _env_type = None

    def __init__(self, name, session_aware=True):
        """
        Constructor for base Environment, should not be instantiated directly.
        """
        if not name:
            raise TypeError("Name is required when creating an Environment.")
        if _sys.version_info.major == 3:
            _raise_error_if_not_of_type(name, [str], 'name')
        else:
            _raise_error_if_not_of_type(name, [str, unicode], 'name')

        self._session = _gl.deploy._default_session
        self.name = name
        self._env_type = type(self).__name__
        self._modified_since_last_saved = None

        if session_aware:
            self._session.register(self)

    def clone(self, name):
        """
        Clones environment and returns the cloned environment.

        Returns
        -------
        out : Environment
            Cloned environment.

        Notes
        -----
        Cloned environment cannot be saved unless it is renamed.
        """
        raise NotImplementedError

    def is_running(self):
        '''
        Returns whether or not the environment actually starts
        '''
        raise NotImplementedError

    def save(self):
        """
        Saves the environment to the session. Can be retrieved using
        gl.deploy.environments[`name`] where `name` is the name of the job.

        See Also
        --------
        graphlab.deploy.environments

        Examples
        --------
        >>> environment.save()
        """
        self._session.save(self)
        return self

    def get_num_workers(self):
        '''
        Return number of workers that can be used to process jobs in parallel
        '''
        return 1

    def __getstate__(self):
        odict = dict.copy(self.__dict__)
        if '_session' in odict:
            del odict['_session']
        return odict

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Environment: [\"name\": %s, \"type\": %s]" % (self.name,
                                                              self._env_type)

    def __to_dict__(self):
        json_dict = {}
        json_dict["name"] = self.name
        json_dict["type"] = self._env_type
        return json_dict


class LocalAsync(_Environment):
    """
    Environment for executing jobs in the background (asynchronously) on your
    local machine.

    Parameters
    ----------
    name : str
        The name for the environment.

    session_aware : str (default True)
        Register this environment with the session (recommended). When
        registered, the environment objects can be retrieved from the session.

    See Also
    --------
    Local

    Examples
    --------
    >>> async = graphlab.deploy.environment.LocalAsync('async')
    """
    _LOCAL_ASYNC_ENV_VERSION = 2

    def __init__(self, name, session_aware=True):
        super(LocalAsync, self).__init__(name, session_aware)

    def __str__(self):
        return "LocalAsync: [\"name\": %s]" % (self.name)

    def _get_version(self):
        return self._LOCAL_ASYNC_ENV_VERSION

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : A GLUnpickler file handle.

        version : Version number.
        """
        # Load the dump.
        obj = unpickler.load()

        # Construct a new object.
        new = cls(obj.name, session_aware = False)
        assert obj._get_version() <= new._get_version()

        # Now copy over the useful parts of the dump.
        lst = ['_env_type']
        _internal_utils.copy_attributes(new, obj, lst)
        return new

    def is_running(self):
        '''
        Returns whether or not the environment actually starts
        '''
        return True

    def clone(self, name):
        """
        Create a safe clone of this object.

        Returns
        -------
        out : Environment
            Cloned environment object.

        Notes
        -----
        - Cloned environment cannot be saved unless it is renamed.
        """
        new = LocalAsync.__new__(LocalAsync)
        new.name = name
        new._modified_since_last_saved = self._modified_since_last_saved
        new._env_type = self._env_type
        new._session = self._session
        return new

class Local(_Environment):
    """
    Environment for executing jobs locally on this machine (not in the background).

    Parameters
    ----------
    name : str
        The name for the environment.

    session_aware : str (default True)
        Register this environment with the session (recommended). When
        registered, the environment objects can be retrieved from the session.

    See Also
    --------
    LocalAsync

    Examples
    --------
    >>> local = graphlab.deploy.environment.Local('local')
    """

    _LOCAL_ENV_VERSION = 2

    def __init__(self, name, session_aware=True):
        super(Local, self).__init__(name, session_aware)

    def __str__(self):
        return "Local: [\"name\": %s]" % (self.name)

    def _get_version(self):
        return self._LOCAL_ENV_VERSION

    @classmethod
    def _load_version(cls, unpickler, version):
        """
        An abstract function to implement save for the object in consideration.

        Parameters
        ----------
        pickler : A GLUnpickler file handle.

        version : Version number.
        """
        # Load the dump.
        obj = unpickler.load()

        # Construct a new object.
        new = cls(obj.name, session_aware = False)
        assert obj._get_version() <= new._get_version()

        # Now copy over the useful parts of the dump.
        lst = ['_env_type', '_modified_since_last_saved',
               '_typename', 'name']
        _internal_utils.copy_attributes(new, obj, lst)
        return new

    def is_running(self):
        '''
        Returns whether or not the environment actually starts
        '''
        return True

    def clone(self, name):
        """
        Create a safe clone of this object.

        Returns
        -------
        out : Environment
            Cloned environment object.

        Notes
        -----
        - Cloned environment cannot be saved unless it is renamed.
        """
        new = Local.__new__(Local)
        new.name = name
        new._modified_since_last_saved = self._modified_since_last_saved
        new._env_type = self._env_type
        new._session = self._session
        return new
