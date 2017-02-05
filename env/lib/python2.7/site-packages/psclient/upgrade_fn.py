import os as _os
import shutil as _shutil
import tempfile as _tempfile
from boto import connect_s3 as _connect_s3
from requests import ConnectionError as _ConnectionError

import psclient.file_util as _file_util
from .predictive_service_client import PredictiveService as _PredictiveService

import logging as _logging
_logger = _logging.getLogger(__name__)

def copy_predictive_object(source_ps, target_ps, source_po_name, target_po_name=None, update=False):
    '''
    Copy a predictive object from a source Predictive Service to a target
    Predictive Service.

    Parameters
    ----------
    source_ps : Predictive Service object
        The source Predictive Service that holds the predictive object specified
        in source_po_name.

    target_ps : Predictive Service object
        The target Predictive Service that will accept the predictive object
        copied from the source Predictive Service.

    source_po_name : str
        The name of the predictive object to be copied. Must exist on the
        source Predictive Service.

    target_po_name : str, optional
        The name of the predictive object to be stored to the target Predictive
        Service. If target_po_name is None, the target Predictive Service would use
        source_po_name as the predictive object name. Default value is None.

    update : boolean, optional
        If a predictive object already exists on the target Predictive Service
        with the name specified by target_po_name, set this to True if you want to
        update the existing predictive object on the target Predictive Service
        with the predictive object from the source Predictive Service. Otherwise,
        leave this to the default value False to prevent update.

    Notes
    -----
    This operation will by-pass `apply_changes` operation on the target Predictive
    Service to add/update the predictive object.

    Examples
    --------
    To copy a predictive object named 'recommender' from a source Predictive
    Service to a target Predictive Service:

        >>> copy_predictive_object(source_ps, target_ps, 'recommender')

    To update the 'recommender' predictive object on the target Predictive Service
    with the 'recommender' predictive object from the source Predictive Service:

        >>> copy_predictive_object(source_ps, target_ps, 'recommender', update=True)

    To copy the 'recommender' predictive object from the source Predictive Service
    to the target Predictive Service and rename it 'rec':

        >>> copy_predictive_object(source_ps, target_ps, 'recommender', 'rec')

    '''
    if not source_ps or type(source_ps) is not _PredictiveService:
        raise ValueError("Invalid source Predictive Service.")
    source_ps._ensure_not_terminated()

    if not target_ps or type(target_ps) is not _PredictiveService:
        raise ValueError("Invalid target Predictive Service.")

    target_ps._ensure_not_terminated()
    target_ps._ensure_no_local_changes()

    # if source is newer than target, fail
    if source_ps.version > target_ps.version:
        raise ValueError("Cannot copy from a version %d Predictive Service " \
                         "to a version %d Predictive Service." % \
                         (source_ps.version, target_ps.version))

    # only support copy with same schema version or from previous version of Predictive Service
    if target_ps.version - source_ps.version > 1:
        raise RuntimeError('Only support copying from version %s or %s of '
            'predictive service. Your source predictive service has version %s' \
            % (target_ps.version - 1, target_ps.version, source_ps.version))

    if source_po_name not in source_ps.get_endpoints():
        raise ValueError("No predictive object named \"%s\" in the source " \
                         "Predictive Service (%s)" % (str(source_po_name), str(source_ps.name)))

    # set the target predictive object name
    target_po_name = source_po_name if not target_po_name else target_po_name

    # get the version for the target predictive service
    if target_po_name in target_ps.endpoints:
        if update is False:
            raise RuntimeError("Cannot update the predictive object %s in the target Predictive Service." \
                            "Please set update to True if you want to update this predictive object in the" \
                            "target Predictive Service." % target_po_name)

        target_version = 1 + target_ps.get_endpoints()[target_po_name]['version']
    else:
        target_version = 1

    # get predictive object info
    source_po_info = source_ps._endpoints[source_po_name]

    po_info = {'version': target_version,
               'docstring': source_po_info['docstring'],
               'cache_state': source_po_info['cache_state'],
               'schema_version': source_po_info['schema_version'],
               'type': source_po_info.get('type', 'model'),
               'description': source_po_info['description'],
               'endpoint_obj': source_po_info.get('endpoint_obj')}

    # get path for predictive objects
    if source_po_info.get('type', 'model') in ['model', 'endpoint']:
        # check if source po is directory or file
        is_dir = source_po_info['schema_version'] >= 3

        source_path = source_ps._get_predictive_object_save_path(source_po_name, source_po_info['version'])
        target_path = target_ps._get_predictive_object_save_path(target_po_name, target_version)

        _copy_predictive_object_files(source_path, target_path, is_dir,
                                    source_ps._environment.aws_credentials,
                                    target_ps._environment.aws_credentials)

    # add po_info to target_ps
    target_ps._endpoints[target_po_name] = po_info

    # save state to s3
    target_ps._save_state()
    try:
        target_ps._environment.poke(target_ps.admin_key)
    except _ConnectionError as e:
        _logger.warn("Unable to connect to target Predictive Service: %s" %
                     (e.message))

    target_ps._update_local_state()
    _logger.info("Successfully copied predictive object \"%s\" from Predictive Service (%s) " \
                 "to Predictive Service (%s)." % (str(source_po_name),
                 str(source_ps.name), str(target_ps.name)))

def _copy_predictive_object_files(source_path, target_path, is_dir, src_credentials, tgt_credentials):
    '''
    Copy either file or folder from source location to target location
    '''
    # Cleanup existing file path if exists
    if _file_util.is_local_path(target_path) and _os.path.exists(target_path):
        _shutil.rmtree(target_path)

    if _file_util.is_s3_path(source_path) and _file_util.is_s3_path(target_path):

        # compare credentials
        _check_aws_credentials(src_credentials, tgt_credentials, source_path)

        # intra s3 copy model
        _file_util.intra_s3_copy_model(source_path, target_path, is_dir, tgt_credentials)
    elif _file_util.is_local_path(source_path):

        _file_util.copy_from_local(source_path, target_path, is_dir = is_dir)

    else:
        tmp_dir = _tempfile.mkdtemp(prefix = 'copy_predictive_object')
        try:
            # download to local first
            local_path = _os.path.join(tmp_dir, 'temp_po_file')
            if _file_util.is_s3_path(source_path):
                _file_util.download_from_s3(
                    source_path,
                    local_path,
                    is_dir=is_dir,
                    aws_credentials=src_credentials,
                    silent=False)
            elif _file_util.is_hdfs_path(source_path):
                _file_util.download_from_hdfs(source_path, local_path, is_dir = False)
            else:
                raise RuntimeError('Unsupported file system type: %s' % source_path)

            # upload from local to remote
            if _file_util.is_s3_path(target_path):
                _file_util.upload_to_s3(local_path, target_path, is_dir=is_dir,
                    aws_credentials=tgt_credentials, silent=False)
            elif _file_util.is_hdfs_path(target_path):
                _file_util.hdfs_mkdir(target_path)
                _file_util.upload_to_hdfs(local_path, target_path, force=True, silent=False)
            else:
                _file_util.upload_to_local(local_path, target_path, is_dir=is_dir, silent=False)

        finally:
            _shutil.rmtree(tmp_dir)

def _check_aws_credentials(src_credentials, tgt_credentials, source_path):
    # check if credentials are the same
    if src_credentials['aws_access_key_id'] == tgt_credentials['aws_access_key_id'] and \
            src_credentials['aws_secret_access_key'] == tgt_credentials['aws_secret_access_key']:
        return

    # make sure tgt_credentials can be used to access source path
    try:
        conn = _connect_s3(**tgt_credentials)
        (bucket_name, s3_directory) = _file_util.parse_s3_path(source_path)
        bucket = conn.get_bucket(bucket_name)
        key = bucket.get_key(s3_directory)
        if not key:
            raise RuntimeError("Unable to find the key within the S3 bucket. Please check your "
                            "aws credentials.")
    except Exception as e:
        raise RuntimeError("Unable to access the correct S3 bucket. Please check"
                                        " your aws credentials. Error: %s" % e)
