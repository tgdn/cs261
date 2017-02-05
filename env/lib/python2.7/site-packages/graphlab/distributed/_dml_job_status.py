import os as _os
import sys as _sys
import logging as _logging
import time
if _sys.version_info.major == 2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

from graphlab.util import file_util as _file_util
from graphlab.extensions import dml_function_invocation as _dml_function_invocation
from ._distributed_shell import JobHandle

# Supported models
from graphlab.toolkits._main import ToolkitError
from graphlab.toolkits.classifier.boosted_trees_classifier import BoostedTreesClassifier as _BoostedTreesClassifier
from graphlab.toolkits.classifier.logistic_classifier import LogisticClassifier as _LogisticClassifier
from graphlab.toolkits.classifier.random_forest_classifier import RandomForestClassifier as _RandomForestClassifier
from graphlab.toolkits.classifier.svm_classifier import SVMClassifier as _SVMClassifier
from graphlab.toolkits.regression.boosted_trees_regression import BoostedTreesRegression as _BoostedTreesRegression
from graphlab.toolkits.regression.linear_regression import LinearRegression as _LinearRegression
from graphlab.toolkits.regression.random_forest_regression import RandomForestRegression as _RandomForestRegression
from graphlab.toolkits.graph_analytics.pagerank import PagerankModel as _PagerankModel
from graphlab.toolkits.graph_analytics.label_propagation import LabelPropagationModel as _LabelPropagationModel

__LOGGER__ = _logging.getLogger(__name__)

_supported_result_objects = {
        'boosted_trees_classifier':_BoostedTreesClassifier,
        'classifier_logistic_regression':_LogisticClassifier,
        'random_forest_classifier':_RandomForestClassifier,
        'classifier_svm':_SVMClassifier,
        'boosted_trees_regression':_BoostedTreesRegression,
        'regression_linear_regression':_LinearRegression,
        'random_forest_regression':_RandomForestRegression,
        'pagerank':_PagerankModel,
        'label_propagation':_LabelPropagationModel,
        }


class DMLJobStatus(object):
    """
    DMLJobStatus tracks a distributed machine learning job while it is running
    on a cluster. It provides status information and the results of the job
    when it has finished. It can also do some limited debugging aid and point
    the user to the log file for them to debug issues. Currently it only
    supports the tracking of distributed training jobs for supervised learning
    models.
    """
    STATE = JobHandle.STATE
    FINAL_STATE = JobHandle.FINAL_STATE

    def __init__(self, model_name, job_handle, working_dir, checkpoint_path=None):
        self._job_handle = job_handle
        self._algo_name = model_name
        self._dml_serdes = _dml_function_invocation()
        self._working_dir = working_dir
        self._checkpoint_path = checkpoint_path
        self._result = None
        self._log_server_address = None
        self._progress_log = ''
        self._log_server_available = True

    def get_results(self):
        """
        Retrieve the result from this job.

        This is a BLOCKING function. It will block until the job is completed
        and a usable result can be returned (usually a trained model). Raises
        an RuntimeError if job fails for any reason, or if the job succeeds but
        the result cannot be returned.

        Returns
        -------
        out : many types
        """
        if self._result is not None:
            return self._result

        (retcode, output) = self._job_handle.wait()
        final_state_ready = False
        while not final_state_ready:
            try:
                state = self.get_final_state()
                final_state_ready = True
            except RuntimeError:
                pass
            time.sleep(1)

        if state == JobHandle.FINAL_STATE.SUCCESS:
            result_location = '/'.join([self._working_dir, 'out'])
            ret_str = _file_util.read(result_location)
            if ret_str is None:
                raise RuntimeError("Failed to read from expected result location: " + str(result_location))
            self._dml_serdes.from_str(ret_str)
            result_dict = self._dml_serdes.to_dict()
            if 'exception' in result_dict:
                __LOGGER__.debug("Found exception in DML result. Result: " + str(result_dict))
                raise ToolkitError(result_dict['exception'])
            if 'result' not in result_dict:
                __LOGGER__.debug("Model not found in DML result. Result: " + str(result_dict))
                raise RuntimeError("Model not found, though job completed successfully!")
            dml_result = result_dict['result']
            result_ctor = _supported_result_objects[self._algo_name]

            self._result = result_ctor(dml_result)
            return self._result
        else:
            self._failed_client_output = output
            __LOGGER__.debug("Client failed. Return code: " + str(retcode) +\
                    "\nSTDOUT:\n" + output[0] + \
                    "\nSTDERR:\n" + output[1])
            __LOGGER__.debug("Current state: " + str(self.get_state()))
            raise RuntimeError("Job failed with final state " + str(state) +\
                    ". Use the status_summary method to diagnose the issue.")

    def get_state(self):
        """
        Checks if this job is currently running.

        Possible answers are ["STATE.PENDING","STATE.RUNNING","STATE.COMPLETED"]

        Returns
        -------
        out : str
        """
        return self._job_handle.get_state()

    def get_final_state(self):
        """
        Checks whether this job succeeded.

        Possible answers are ["FINAL_STATE.SUCCESS","FINAL_STATE.FAILURE","FINAL_STATE.KILLED"]

        Raises RuntimeError if the job is still running, and ValueError if the
        job scheduluer supplies unexpected output.

        Returns
        -------
        out : str
        """
        return self._job_handle.get_final_state()

    def get_log_file_path(self):
        """
        Returns the file path of the execution log for this job.

        This is a temporary storage of the job's log file according to the job
        scheduler. The file is deleted when this object gets deleted. If the
        log file's contents are needed past the life of this object, please
        copy it to a permanent location.

        Returns
        -------
        out : str
        """
        return self._job_handle.get_log_file_path()

    def last_checkpoint(self):
        """
        Gives the name of the last checkpoint written, if applicable.

        This is intended to be used with models that support the
        "model_checkpoint_path" argument. The value returned from this function
        can be directly supplied to the "resume_from_checkpoint" argument of
        the create method that returned this object.

        Raises if no checkpoints found or this job does not train a model that
        writes checkpoints during training.

        Returns
        -------
        out : str
        """
        if self._checkpoint_path is None:
            raise TypeError("Job does not save checkpoints.")
        matching_files = _file_util.find(self._checkpoint_path, "model_checkpoint_[0-9]+")
        max_checkpoint = -1
        max_checkpoint_name = None
        for name in matching_files:
            iteration = name.split('_')[-1]
            try:
                iteration = int(iteration)
            except ValueError:
                # Shouldn't happen given our pattern anyways
                continue
            if max_checkpoint < iteration:
                max_checkpoint = iteration
                max_checkpoint_name = name

        if max_checkpoint == -1:
            raise RuntimeError("No checkpoints found in " + self._checkpoint_path)

        return '/'.join([self._checkpoint_path,max_checkpoint_name])

    def summary(self):
        """
        A human-readable summary of the job's execution.

        A summary is only available after the job has completed. Output is
        printed, not returned.

        Returns
        -------
        None
        """
        print(self._get_summary())
        print("\nIf you need to dig further, use the get_log_file method.")

    def get_progress(self):
        """ Get progress log of the job
        Returns
        -------
        out : str
            Progress log of the job
        """
        state = self.get_state()
        if state == JobHandle.STATE.PENDING:
            __LOGGER__.warn('Job has not started')
        elif state == JobHandle.STATE.RUNNING:

            if self._log_server_available:
                try:
                    diff = self._get_progress_from_server()
                    self._progress_log += diff
                except Exception as e:
                    __LOGGER__.warn(e)
                    self._log_server_available = False

            if not self._log_server_available:
                self._progress_log = self._get_progress_from_file()

        elif state == JobHandle.STATE.COMPLETED:
            self._progress_log = self._get_progress_from_file()
        else:
            pass
        return self._progress_log

    def _get_progress_from_file(self):
        log_location = '/'.join([self._working_dir, 'progress.log'])
        if _file_util.exists(log_location):
            content = _file_util.read(log_location)
            if content:
                return content[:-1]
        else:
            __LOGGER__.warn('Progress log file cannot be found')
        return ""

    def _get_progress_from_server(self):
        url = self._get_job_log_server_address()
        if url:
            content = urlopen(url, timeout=3).read()
            if len(content):
                return content
        else:
            __LOGGER__.warn('Progress log server is not available')
        return ""

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self._job_handle.__str__()

    def _get_summary(self):
        return self._job_handle.get_diagnosis()

    def _get_job_log_server_address(self, timeout=10):
        if self._log_server_address:
            return self._log_server_address
        log_server_address_file = '/'.join([self._working_dir, 'metric_server_address'])
        starttime = time.time()
        timeout = False
        __LOGGER__.info('Waiting for log server address to be available')
        while(not _file_util.exists(log_server_address_file)):
            time.sleep(1)
            if (time.time() - starttime) > timeout:
                __LOGGER__.info('Timeout waiting for log server address')
                timeout = True
                break
        if not timeout:
            ret_str = _file_util.read(log_server_address_file)
            if ret_str.endswith('$'):
                self._log_server_address = ret_str[:-1] + "/progress"
                __LOGGER__.info('Log server address: %s' % self._log_server_address)
                return self._log_server_address
        return ""
