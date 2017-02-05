import os as _os
import re as _re
import sys as _sys
import fnmatch as _fnmatch
import logging
import shelve as _shelve
import six as _six
from graphlab.util import lockfile as _lockfile
import datetime as _datetime

import graphlab as _gl
from . import _artifact
from . import _internal_utils
import shutil as _shutil
import glob as _glob

from graphlab.util import _make_temp_filename

__LOGGER__ = logging.getLogger(__name__)

# Version
# -----------
# GLC 1.2 = 1
# GLC 1.3 = 2
# GLC 1.4 = 3
_SESSION_VERSION = 3

SEPARATOR = " \001\002\003\004\005 "
class Key(str):
    """
    The python module 'shelve' implements a disk backed dictionary. This requires
    string keys but can store objects of all types. We are using shelve to
    manage the Scoped-Session.

    Each Key in the Scoped-Session is the name of the Artifact in that session.
    The global key of the object is a combination of

    "(Name) (separator) (Typename)" where:

    (Name)      : Name of the artifact
    (Separator) : A string separator
    (Type)      : The type of the artifact (Task | Job | Env)
    """

    # We use a random magic string for a separator because we would like to be
    # able to separate the index into type and name by simply separating on the
    # key.
    def __init__(self, from_str):
        self.name, self.typename = from_str.split(SEPARATOR)

    @classmethod
    def from_tuple(cls, name, typename):
        """ Return a Key object from a tuple of (name, typename)
        """
        return cls("%s%s%s" % (name, SEPARATOR, typename))

    def to_tuple(self):
        return (self.name, self.typename)

    def __str__(self):
        return "%s%s%s" % (self.name, SEPARATOR, self.typename)

    def __repr__(self):
        return "Key(%s, %s)" % (self.name, self.typename)


class Session(object):
    """
    Manages GraphLab Artifacts which include:

    (1) Jobs,
    (2) Environments,
    (3) Tasks, and

    A session is not intended to be interacted with directly, instead use
    graphlab.deploy.jobs or graphlab.deploy.environments.
    """

    def __init__(self, location):
        """
        Creates a new session, specifying a directory to use for the session.

        Parameters
        ----------
        location: os.path
            An absolute path which is used as the home directory for managing
            the session.

        Notes
        -----
        - This directory will be used for saving and loading all Artifacts.

        - If the directory does not exist it will be created.
        """

        if location is None:
            raise Exception('Location is a required parameter.')
        self.location = location

        __LOGGER__.debug('Using session dir %s for Jobs and Environments.' % self.location)

        # Validate location.
        if _os.path.exists(location):
            if _os.path.isfile(location):
                raise Exception('Illegal location for session, must be a directory.')

        # Directory does not exist. Create it.
        else:
            try:
                _os.makedirs(self.location)
            except OSError as e:
                self.location = _make_temp_filename(prefix='tmp_session_')
                if not _os.path.exists(self.location):
                    _os.makedirs(self.location)
                __LOGGER__.warning(("Unable to create session in specified "
                   "location: '%s'. Using: '%s'") % (location, self.location))


        # Locking object for concurrent processes.
        self._lock = _lockfile.LockFile(_os.path.join(self.location, 'lock'))

        # Find the version file for the session to open it.
        version_file = _os.path.join(self.location, 'version')

        with self._lock:
            if not _os.path.exists(version_file):
                version = _SESSION_VERSION
            else:
                try:
                    version = int(open(version_file).read().strip())
                except Exception as e:
                    __LOGGER__.warning(
                        "Corruption in Graphlab Create artifacts directory:"
                        " Bad version file (%s). (Trace %s)" % (version_file, e))
                    version = _SESSION_VERSION

        # if GLC-1.3.
        if version == 2:
            archive_dir = _os.path.expanduser(_os.path.join('~',
                                         '.graphlab', 'artifacts-archive'))

            # Find a new directory to move all the files over to.
            ind = 1
            with self._lock:
                while _os.path.exists("%s-%s" % (archive_dir, ind)):
                    ind = ind + 1
                archive_dir = "%s-%s" % (archive_dir, ind)
                _os.makedirs(archive_dir)

            # Warn the user.
            __LOGGER__.info(
             "Jobs and environments saved using GraphLab Create v1.3 cannot be loaded using latest GraphLab Create release.")
            __LOGGER__.info(
            'Archiving all your metadata in {path}. Load them in GraphLab Create v1.3 by setting'\
            ' GRAPHLAB_ARTIFACTS_TMP_SESSION_DIR={path}'
                    .format(path = archive_dir))

            # Move all the files over to the new directory.
            for f in _glob.glob(_os.path.join(self.location, '*')):
                if f != _os.path.abspath(self._lock.lock_file):
                    rel_dst = _os.path.relpath(f, self.location)
                    abs_dst = _os.path.join(archive_dir, rel_dst)
                    _shutil.move(f, abs_dst)

        # Make a results directory.
        results_dir = '/'.join([self.location, "results"])
        if not _os.path.isdir(results_dir):
            with self._lock:
                if not _os.path.isdir(results_dir):
                    _os.makedirs(results_dir)
        self.results_dir = results_dir

        # The cache for all unsaved objects.
        self._unsaved_objects = {}

        # A disk-backed index file to maintain state for all artifacts. This
        # allows multiple IPython sessions to connect to a single index file.
        self._index_file = _os.path.join(self.location, 'index')

        # Write a version file in the directory only if the version has changed.
        if not _os.path.exists(version_file):
            with self._lock:
                with open(version_file, 'w') as v_file_handle:
                    v_file_handle.write(str(_SESSION_VERSION))

    def exists(self, name, typename):
        """"Returns True if an object of a given type with given name exits in
        current session, False otherwise

        Parameters:
        ------------
        name : str
            Name of the artifact

        typename: str
            Type name of the artifact

        """
        key = Key.from_tuple(name, typename)
        return not self._is_unique(key)

    def contains(self, artifact):
        """ True if the Session contains an object.

        Parameters:
        ------------
        artifact : Artifact
            A Graphlab artifact object.

        """
        key = Key.from_tuple(artifact.name, artifact._typename)
        return not self._is_unique(key)

    def _is_unique(self, key):
        """
        Check if the Key(name, typename) is unique.
        """
        if key in self._unsaved_objects:
            return False

        # Lock the index file
        ret = True
        with self._lock:
            index = _shelve.open(self._index_file)
            if key in index:
                ret = False
            index.close()
        return ret

    def register(self, artifact):
        """
        Objects that should be managed by this session are required to register
        with the session.

        Note: Register does not persist the object. Use save for persisting.

        Parameters
        ----------
        artifact : Artifact
            Two artifacts of the same type, cannot share the same name.

        Returns
        -------
        True if successfully registered, False otherwise
        """
        if artifact is None:
            return False

        key = Key.from_tuple(artifact.name, artifact._typename)
        if not key.name:
            __LOGGER__.warning("A name is required for objects managed by a session.")
            return False

        if self._is_unique(key) == False:
            raise KeyError(("An item with the name '%s' already exists,"
                                " please rename the item.") % artifact.name)

        if not artifact._session_registration_date:
            artifact._session_registration_date = _datetime.datetime.now()
        self._unsaved_objects[key] = artifact
        return True

    def list(self, typename = None):
        """
        List all items managed by this session.

        Parameters
        ----------
        typename : str, optional

            Optionally limit the output a specific type.

        Returns
        -------
        SFrame with columns for name, type, Modified?, In-Memory?, Unsaved?
        """

        display_all = True if typename is None else False
        values = []
        # List items in unsaved objects.
        for k, value in _six.iteritems(self._unsaved_objects):
            if Key(k).typename == typename or display_all:
                values.append(value._get_metadata())

        # List items in the index file.
        with self._lock:
            index = _shelve.open(self._index_file)
            for k, value in _six.iteritems(index):
                if Key(k).typename == typename or display_all:
                    del value['filename']
                    values.append(value)
            index.close()

        # Return as SFrame
        if values:
            sf = _gl.SFrame(values).unpack('X1', column_name_prefix='')\
                                   .sort('Creation date', ascending = False)
            # Put creation date in the end (it looks ugly otherwise.)
            sa = sf['Creation date']
            sf.remove_column('Creation date')
            sf = sf.add_column(sa, name='Creation date')

            # Add an index orderd by datetime.
            sf = sf.add_row_number(column_name='Index')
            return sf
        else:
            return _gl.SFrame()

    def _delete_key_from_index(self, key, index=None, silent=False):
        """
        Helper function to delete key from index file. Returns True if something
        was deleted. False otherwise.

        Parameters
        ----------
        key : str
            Key (as a string) to be deleted from the index file.

        silent: bool
            When True, do not display a log message for each object deleted.

        Notes
        -----
        - This operation is not locked. Caller must acquire lock.
        """
        new_index = index is None
        if new_index:
            index = _shelve.open(self._index_file)
        ret = False
        if key in index:
            job_path = _os.path.join(self.location, index[str(key)]['filename'])
            del index[str(key)]

            if _os.path.isdir(job_path):
                _shutil.rmtree(job_path)
            else:
                _os.remove(job_path)

            if not silent:
                __LOGGER__.info("Deleted reference to %s('%s') from current session." % (Key(key).typename,
                                                                                Key(key).name))
            ret = True
        if new_index:
            index.close()
        return ret

    def _delete_key_from_unsaved_objects(self, key, silent=False):
        """
        Helper function to delete key from unsaved objects. Returns True if something
        was deleted. False otherwise.

        Parameters
        ----------
        key : str
            Key (as a string) to be deleted from the index file.

        silent: bool
            When True, do not display a log message for each object deleted.
        """
        if key in self._unsaved_objects:
            del self._unsaved_objects[key]
            if not silent:
                __LOGGER__.info("Deleted reference to %s('%s') from current session." % (Key(key).typename,
                                                                                Key(key).name))
            return True
        return False

    def delete(self, artifact, typename, pattern=None, silent = False):
        """
        Delete artifact from disk.

        Warning: The following operation will permanently delete the object
        from the file-system and will not ask for permission to do so.

        Parameters
        ----------
        artifact : object | str | int
            Artifact to delete, passed as an object, or name, or index.

        typename : str
            Limit the output a specific type.

        pattern : str
            Delete a collection of artifacts ex. "*-Local-*" that match a pattern
            string (uses fnmatch.fnmatch to match the patterns).

        silent: bool
            When True, do not display a log message for each object deleted.
        """
        # Index lookup
        if isinstance(artifact, int):
            obj = self._get_name_from_row_id(artifact, typename)
        else:
            obj = artifact

        # Not a pattern. Search for the key to delete.
        if pattern is None:
            name = obj if type(obj) == str else obj.name
            key = Key.from_tuple(name, typename)

            # Delete cache.
            del_unsaved = self._delete_key_from_unsaved_objects(key, silent)
            # Delete from index.
            if not del_unsaved:
                with self._lock:
                    del_ind = self._delete_key_from_index(key, silent=silent)

            # Raise error if not found.
            if not (del_unsaved or del_ind):
                raise KeyError("No object found with key '%s'." % name)

        # Match patterns
        else:
            # Remove persisted objects.
            with self._lock:
                index = _shelve.open(self._index_file)
                for key in index.keys():
                    _key, _typename = Key(key).to_tuple()
                    if _fnmatch.fnmatch(_key, pattern) and _typename == typename:
                        self._delete_key_from_index(key, index)
                index.close()

            # Remove from cache.
            delete = []
            for key in self._unsaved_objects:
                _key, _typename = Key(key).to_tuple()
                if _fnmatch.fnmatch(_key, pattern) and _typename == typename:
                    delete.append(key)
            for i in delete:
                self._delete_key_from_unsaved_objects(i)

    def _get_name_from_row_id(self, index, typename):
        output_sf = self.list(typename=typename)
        if index >= output_sf.num_rows():
            raise IndexError("No object found with index %s." % index)
        return output_sf['Name'][index]

    def load(self, name, typename):
        """
        Load an item by name and type.

        Parameters
        ----------
        name : str | int
            Name of item to load. The name of object or the index can be
            used to load the object.

        typename : str
            Limit the output a specific type.

        Returns
        -------
        Instance of the item.
        """

        # Key not found error message.
        if typename is None:
            raise KeyError("Required type to load object.")

        # Index lookup
        if isinstance(name, int):
            name = self._get_name_from_row_id(name, typename)

        # Check in cache.
        key = Key.from_tuple(name, typename)
        if key in self._unsaved_objects:
            return self._unsaved_objects[key]

        # Not in unsaved?: Get the filename from the index.
        filename = None
        with self._lock:
            index = _shelve.open(self._index_file)
            if key in index:
                filename = _os.path.join(self.location, index[str(key)]['filename'])
            index.close()

        # Now load the object.
        if filename:
            obj = _artifact.load_artifact(filename)
            obj.session = self
            return obj
        raise KeyError("No object found with key '%s'" % key.name)

    def _get_filename_from_name(self, name):
        """Gets a valid filename from the object name."""
        if name is None or name == "":
            raise AttributeError('Missing name')
        name = str(name).strip().replace(' ', '_')
        return _re.sub(r'(?u)[^-\w.]', '', name)

    def save(self, obj, typename = None):
        """
        Save the item to this session.

        Parameters
        ----------
        obj : object
            Object to save to this session.
        """

        if not Session._is_known_type(obj):
            raise Exception("Trying to save an object of unknown type.")

        name = obj.name
        typename = obj._typename if not typename else typename

        savedir = self.location
        if _internal_utils.is_gl_artifact_class(obj.__class__):
            key = Key.from_tuple(name, typename)
            if not obj._session_registration_date:
                obj._session_registration_date = _datetime.datetime.now()

            filename = "%s.%s" % (self._get_filename_from_name(obj.name), typename)
            obj._modified_since_last_saved = False
            # Persist the object.
            with self._lock:
                obj._save_to_file(_os.path.join(savedir, filename))

                # Remove from unsaved objects (only if it exists)
                self._delete_key_from_unsaved_objects(key, silent=True)

                # Update the index file with the object metadata.
                metadata = obj._get_metadata()
                metadata.update({'filename': filename})

                # Update the shelve.
                index = _shelve.open(self._index_file)
                index[str(key)] = metadata
                index.close()
            return

    def __getitem__(self, item):
        return self._load(item)

    def __delitem__(self, key):
        return self._delete(key)

    @staticmethod
    def _is_known_type(obj):
        """
        Helper method that returns True when type is supported by Session.
        """
        return _internal_utils.is_gl_artifact_class(obj.__class__)

    def __repr__(self):
        return self.__str__(typename=None)

    def __str__(self, typename=None):
        if typename is None:
            ret = "Session (location: %s)\n" % (_os.path.abspath(self.location))
        else:
            ret = "%s(s): \n" % typename

        output_sf = self.list(typename=typename)
        if output_sf.num_rows() == 0:
            ret += 'Nothing to display.'
        else:
            rows_of_table = output_sf.__get_pretty_tables__(max_column_width=60)
            ret += '\n'.join([str(tb) for tb in rows_of_table])
        return ret


def _open(location=None):
    """
    Opens and returns an Artifact session compatible with the most recent version.

    If the session opened is from an older version, the Session constructor
    will upgrade the version of the Session.

    Parameters
    ----------
    location : os.path

        Disk location for this session, needs to be a directory. Default value
        is ~/.graphlab/artifacts.

    """
    if location is None:
        # Keep artifacts folder unchanged for Python2 for backward compatibility.
        # For Python version greater than 2, use 'artifacts<version-number>' as the folder
        if _sys.version_info.major > 2:
            location = _os.path.expanduser(_os.path.join('~', '.graphlab',
                                        'artifacts%d' % _sys.version_info.major))
        else:
            location = _os.path.expanduser(_os.path.join('~', '.graphlab', 'artifacts'))

    if 'GRAPHLAB_ARTIFACTS_TMP_SESSION_DIR' in _os.environ:
        location = _os.environ['GRAPHLAB_ARTIFACTS_TMP_SESSION_DIR']
        __LOGGER__.info('Using session dir: %s' % location)

    return Session(location)


class ScopedSession(object):
    def __init__(self, session, type, docstring):
        self._session = session
        self._typename = type.__name__.split('_')[-1]
        self.__doc__ = docstring

    def __repr__(self):
        return self._session.__str__(typename=self._typename)

    def list(self):
        """
        Return an SFrame that contains a list of objects in the workbench.

        Returns
        -------
        out : SFrame
            An SFrame describing a list of objects type with relevant metadata.

        See Also
        --------
        delete, show
        """
        return self._session.list(typename=self._typename)

    def show(self):
        """
        Show all objects managed by the session in Canvas.

        Returns
        -------
        view: graphlab.canvas.view.View
            An object representing the GraphLab Canvas view

        See Also
        --------
        list

        """
        if self._typename == 'Job':
            import graphlab.canvas
            import graphlab.canvas.views.job_session
            return graphlab.canvas.show(
                  graphlab.canvas.views.job_session.ScopedSessionView(self))
        return 'Canvas visualization is not supported for typename %s.' % self._typename

    def delete(self, item=None, pattern=None, silent=False):
        """
        Delete an item from local disk. For remote objects, this operation
        does not halt remote execution (see Notes for details).

        Parameters
        ----------
        item : object | str | int
            Item to delete, passed as an object, by name, or by index.

        pattern : str
            A pattern string to delete objects in the workbench. For example,
            `*-Local-*` deletes all objects with `-Local-` in the name.
            The pattern match is made using fnmatch.fnmatch.

        silent : bool
            When True, do not log messages for each object deletion call.

        See Also
        --------
        list, show

        Notes
        -----
        - For remotely running Jobs, this operation deletes the locally saved
          endpoint to the job, and will NOT cancel the job.
        - Please exercise caution while deleting objects because this will not
          ask permission and will delete from the file system.

        """
        return self._session.delete(item, pattern=pattern,
                                    typename=self._typename, silent = silent)

    def __getitem__(self, item):
        return self._session.load(item, typename=self._typename)

    def __delitem__(self, key):
        self._session.delete(key, typename=self._typename)
        return

    def __contains__(self, item):
        self._session.contains(item)
