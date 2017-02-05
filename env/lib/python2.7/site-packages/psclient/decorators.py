def required_packages(packages):
    """
    Decorator to annotate the set of packages required for executing a custom
    function.

    Parameters
    ----------
    packages: list[str]
        A list that specifies python packages required by the decorated function.
        The dependencies are specified in the format of local file paths or
        distutils, like: ['mysql==0.3.0', 'abc==1.2.3'].
        The package can either be a pypi package, or a local Python egg.
        The egg file has to be installable in a Linux Operating System.

    Returns
    ----------
    A decorator specifying required packages.

    See Also
    --------
    required_files

    Examples
    --------
    This example specifies the names package as a dependency for a custom
    function:

    .. sourcecode:: python

        @required_packages(['names==0.3.0'])
        def my_custom_query(input):
            import names
            return [names.get_first_name() for i in range(input['size'])]

        >>> ps.deploy('my_custom_query_name', my_custom_query)

    To add a local Python egg as dependent package, use its local file path:

    .. sourcecode:: python

        @required_packages(['~/my_package_folder/package1.tar.gz'])
        def my_function():
            import package1
            ...

    You can mix the local file and the pypi package specification:

    .. sourcecode:: python

        @required_packages(['~/my_package_folder/package1.tar.gz', 'names==0.3.0'])
        def my_function():
            import package1
            import names
            ...

    """
    def decorator_without_arguments(func):
        # validate python package dependencies, should be a list of strings
        if not isinstance(packages, list) or \
            any([not isinstance(dependency, str) for dependency in packages]):
            raise TypeError(("python package dependencies has to be a list of"
                      "strings like: ['mysql==1.0.0', 'another==0.2.3']"))
        func.func_dict['required_packages'] = packages
        return func
    return decorator_without_arguments

def required_files(files, pattern='*'):
    ''' Decorator to annotate the set of dependent file(s) required for
    executing a custom predictive object.

    Package required files for remote execution. This ensures that code written
    locally can safely run in remote environments. The files will be packaged
    and shipped to the remote execution environment. The files will be laid
    out in the same directory structure, in the remote machine, as was present
    in the current working directory.

    Parameters
    ----------
    files : list[str] | str
        Files can be one of the following types:

        - If 'files' is a string and points to a directory, then all files under
          the directory with the given pattern will be shipped.
        - If 'files' is a string and points to a file, the only the one file is
          shipped, pattern will be ignored.
        - If 'files' is a list of string, then we treat them as a set of files to
          be shipped.

    pattern: str
        The file name pattern, it is used as a filter to filter out files that
        are not needed.

    See Also
    --------
    required_packages

    Examples
    --------

        To include all files in a given directory.

        .. sourcecode:: python

            @required_files('my_module_directory_name', pattern='*.py')
            def my_function(input):
                # my logic
                return output

        To include a list of files required by the function.

        .. sourcecode:: python

            @required_files(['file1.py', 'file2.py'])
            def my_function(input):
                # my logic
                return output

        To include a mix of files and directories of files.

        .. sourcecode:: python

            @required_files(['file1.py', 'folder1'], pattern='*.py')
            def my_function(input):
                # my logic
                return output

    Notes
    -----
    - Note that all files are going to be recorded as a relative path to the
      current working directory. When shipped to remote machine, the files
      are going to be laid out in exactly the same structure as in your folder.
    '''

    import os as _os

    def _find_files(directory, pattern):
        '''get all files in a given directory with given pattern'''
        import fnmatch as _fnmatch
        for root, dirs, files in _os.walk(directory):
            for basename in files:
                if _fnmatch.fnmatch(basename, pattern):
                    filename = _os.path.join(root, basename)
                    yield filename

    def _read_file_or_directory(base_dir, file_or_dir):
        if _os.path.isfile(file_or_dir):
            relpath = _os.path.relpath(file_or_dir, start=base_dir)
            if '..' in relpath:
                suggested_path = _os.path.split(file_or_dir)[0]
                raise ValueError(\
                    "We use the file's relative path to your current working directory '%s' to"
                    " determine your file's remote location. '%s' is in a path that is not"
                    " under current working directory. Please switch your current working"
                    " directory using os.chdir() to switch to a parent folder of your files"
                    " to be uploaded and then try again."
                    " For example, you may do:\n\n"
                    " \timport os\n"
                    " \tos.chdir('%s')\n" % (base_dir, file_or_dir, suggested_path))
            return {relpath: file_or_dir}
        elif _os.path.isdir(file_or_dir):
            ret = {}
            for f in _find_files(file_or_dir, pattern):
                ret.update(_read_file_or_directory(base_dir, f))
            return ret
        else:
            raise TypeError('"%s" is not a file or directory' % file_or_dir)

    if isinstance(files, basestring):
        files = _os.path.realpath(_os.path.expanduser(files))
        if not _os.path.isdir(files) and not _os.path.isfile(files):
            raise TypeError('"%s" does not point to any file or directory' % files)

        files = [files]

    if not isinstance(files, list) or \
        not all([isinstance(f, basestring) for f in files]):
        raise TypeError('"files" parameter has to be either a file name, directory' \
            ' name or a list of files')

    all_files = {}
    for f in files:
        f = _os.path.realpath(_os.path.expanduser(f))
        all_files.update(_read_file_or_directory(_os.getcwd(), f))

    def decorator_without_arguments(func):
        func.func_dict['required_files'] = all_files
        return func

    return decorator_without_arguments
