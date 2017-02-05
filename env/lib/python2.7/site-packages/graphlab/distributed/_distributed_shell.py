import logging
import subprocess
import re
import os
import tempfile
import shutil

from graphlab.util import subprocess_exe
from _yarn_client import YarnClient

logger = logging.getLogger(__name__)
_script_dir = os.path.dirname(os.path.realpath(__file__))

def _yarn_exit_code_to_reason(exit_code):
    if exit_code == 0:
        return "Succeeded."
    elif exit_code == -100:
        return "Container killed by framework, either due to being "\
        "released by the application or being 'lost' due to node failure."
    elif exit_code == -101:
        return "Disk failed on node."
    elif exit_code == -102:
        return "Container preempted by framework."
    elif exit_code == -103:
        return "Container terminated due to exceeding allocated virtual memory."
    elif exit_code == -104:
        return "Container terminated due to exceeding allocated physical memory."
    elif exit_code == -105:
        return "Container terminated by stop request by application master."
    elif exit_code == -106:
        return "Container terminated by resource manager."
    elif exit_code == -107:
        return "Container terminated after application finished."
    else:
        return "Unknown: Invalid exit code."

class JobHandle(object):
    class STATE:
        PENDING = "STATE.PENDING"
        RUNNING = "STATE.RUNNING"
        COMPLETED = "STATE.COMPLETED"

    class FINAL_STATE:
        SUCCESS = "FINAL_STATE.SUCCESS"
        FAILURE = "FINAL_STATE.FAILURE"
        KILLED = "FINAL_STATE.KILLED"

    def get_state(self):
        """
        Get the state of the job
        """
        pass

    def get_final_state(self):
        """
        Get the final state of the job
        """
        pass

    def get_diagnosis(self):
        """
        Return diagnosis string of the job
        """
        pass

    def get_log_file_path(self):
        """
        Return path to the log_file from the completed job. Job state must be COMPLETED.
        """
        pass

    def wait(self):
        """
        Block until the job state is COMPLETED
        """
        pass

    def cancel(self):
        """
        Request job to cancel. Block until job state is COMPLETED.
        """
        pass


class HadoopJobHandle(JobHandle):
    #### START of public interface ####
    def __init__(self, env, app_info):
        self.job_id = app_info['app_id']
        self._client = YarnClient(env)
        self._proc = app_info['proc_object']
        self._client_rc = None
        self._yarn_log_file_path = None
        self._final_yarn_state = None
        self._script_file = app_info['script_file']

    def __del__(self):
        try:
            if self._yarn_log_file_path:
                log_temp_dir = os.path.dirname(self._yarn_log_file_path)
                shutil.rmtree(log_temp_dir, ignore_errors=True)
            os.remove(self._script_file)
        except Exception:
            pass

    def get_state(self):
        yarn_state = self._client.get_state(self.job_id)['state']
        if yarn_state in ["NEW", "NEW_SAVING", "SUBMITTED", "ACCEPTED"]:
            return self.STATE.PENDING
        elif yarn_state in ["RUNNING"]:
            return self.STATE.RUNNING
        elif yarn_state in ["FINISHING", "FINISHED", "FAILED", "KILLED"]:
            return self.STATE.COMPLETED

        raise ValueError("Received unexpected state: " + str(yarn_state))

    def get_yarn_state(self):
        """
        Returns the state of the job as reported by YARN.
        """
        return self._client.get_state(self.job_id)['state']

    def get_final_state(self):
        app_info = self._client.get_app_info(self.job_id)
        final_state = app_info['app']['finalStatus']
        if final_state == 'UNDEFINED':
            raise RuntimeError("Final state not currently available.")
        elif final_state == 'SUCCEEDED':
            return self.FINAL_STATE.SUCCESS
        elif final_state == 'FAILED':
            return self.FINAL_STATE.FAILURE
        elif final_state == 'KILLED':
            return self.FINAL_STATE.KILLED
        else:
            raise ValueError("Received unexpected final state: " + str(final_state))

    def get_diagnosis(self):
        """
        Returns a human-readable, high-level summary of the job post-execution.
        """
        ret_str = ""
        run_info = self._extract_info_from_yarn_log()
        ret_str += "Container list: " + str(run_info['container_list']) + "\n\n"
        for con in run_info['container_list']:
            post_mortem_status = run_info['post_mortem']
            ret_str += str(con) + ":" + "\n"
            if con in post_mortem_status:
                code = post_mortem_status[con]['exit_status']
                ret_str += "  " + _yarn_exit_code_to_reason(code) + " (" + str(code) + ")" + "\n"
            elif con == run_info['application_master']:
                ret_str += "  Application Master\n"
            else:
                ret_str += "  Report not found in log.\n"

        return ret_str

    def get_log_file_path(self):
        """
        Return file path of temporary file holding YARN log data.
        """
        fpath = self._get_yarn_log_file_path()
        if not fpath:
            raise RuntimeError('Yarn log file is not available yet. The application may still be running.')
        return fpath

    def wait(self):
        """
        Wait until the YARN client process has exited.
        """
        if self._client_rc is None:
            data = self._proc.communicate()
            self._client_rc = self._proc.returncode
            self._client_output = data
        return (self._client_rc, self._client_output)

    #### END of public interface ####

    #### START of magic method implementations ####
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self._client_rc is None:
            return ("Job submitted with application_id: " + str(self.job_id))
        else:
            return self.get_diagnosis()

    #### END of magic method implementations ####

    #### START of private methods ####
    def _get_yarn_log_file_path(self):
        """
        Copies the contents of the YARN log file for this job in to a temporary
        file. This file gets deleted when this object gets deleted.
        """
        if self._yarn_log_file_path is not None:
            return self._yarn_log_file_path
        self._log_temp_dir = tempfile.mkdtemp()
        args = ['logs', '-applicationId', self.job_id]
        sub_ret = subprocess_exe('yarn', args, out_log_prefix=os.path.join(self._log_temp_dir, "tmp_log"))

        if sub_ret['success'] == False:
            if sub_ret['python_exception'] is not None:
                raise sub_ret['python_exception']
            else:
                # Apparently this application returns errors on stdout
                with open(sub_ret['stdout'], 'r') as f:
                    raise RuntimeError("Failed to retrieve application log from YARN: " + ''.join(f))
        self._yarn_log_file_path = sub_ret['stdout']
        return sub_ret['stdout']

    # Only works for log aggregation
    def _extract_info_from_yarn_log(self):
        """
        Parses YARN log for diagnostic codes and any other relevant information.
        Since it reads the log file it is only useful for post-mortem information.
        """
        fpath = self._get_yarn_log_file_path()
        container_regex = re.compile("Container: (?P<container_id>container_.*[0-9]) on .*")
        diag_regex = re.compile(".*Got container status for containerID=(?P<container_id>.*), state=(?P<state>.*), exitStatus=(?P<exit_status>.*), diagnostics=(?P<diagnostics>.*)$")
        log_info = {}
        post_mortem_state = {}
        log_info['container_list'] = []
        min_id_num = 9999999999999999
        log_info['application_master'] = None
        id_num = None
        with open(fpath, 'r') as f:
            for line in f:
                cont_match = container_regex.match(line)
                if cont_match:
                    container_id = cont_match.group('container_id')
                    log_info['container_list'].append(container_id)
                    id_num = int(container_id.split('_')[-1])
                    if id_num < min_id_num:
                        min_id_num = id_num
                        log_info['application_master'] = container_id
                diag_match = diag_regex.match(line)
                if diag_match:
                    if diag_match.group('state') == 'COMPLETE':
                        cur_cid = diag_match.group('container_id')
                        post_mortem_state[cur_cid] =\
                            {
                             'exit_status':int(diag_match.group('exit_status')),
                             'diagnostics':diag_match.group('diagnostics'),
                            }
        log_info['container_list'] = sorted(log_info['container_list'])
        log_info['post_mortem'] = post_mortem_state

        return log_info

    #### END of private methods ####


class HadoopDistributedShell(object):
    YARN_APPLICATION_JAR = 'graphlabDistributedShell-1.0.jar'

    def __init__(self, hadoop_cluster):
        """
        Creates a HadoopDistributedShell
        """
        self.hadoop_cluster = hadoop_cluster
        _script_dir = os.path.dirname(os.path.realpath(__file__))
        self.application_jar = os.path.join(_script_dir, self.YARN_APPLICATION_JAR)

    def run(self, app_name, script_file, num_nodes):
        """
        Launch distributed shell with script file and number of nodes
        """
        app_info = self._launch_with_java_client(app_name, script_file, num_nodes)
        return HadoopJobHandle(self.hadoop_cluster, app_info)

    def _launch_with_java_client(self, appname, script_file, num_nodes):
        """
        Launches the graphlabDistributedShell client jar in a subprocess with the appropriate arguments.
        """
        subproc_args = []
        if self.hadoop_cluster.hadoop_conf_dir is not None:
            subproc_args.append("--config")
            subproc_args.append(str(self.hadoop_cluster.hadoop_conf_dir))
        subproc_args.append("jar")
        subproc_args.append(self.application_jar)
        subproc_args.append("-jar")
        subproc_args.append(self.application_jar)
        subproc_args.append("-appname")
        subproc_args.append(appname)
        subproc_args.append("-shell_script")
        subproc_args.append(str(script_file))
        subproc_args.append("-num_containers")
        subproc_args.append(str(num_nodes))
        subproc_args.append("-container_vcores")
        subproc_args.append(str(self.hadoop_cluster.num_vcores))
        subproc_args.append("-container_memory")
        subproc_args.append(str(self.hadoop_cluster.container_size))

        logger.info('Launch Yarn application: hadoop %s' % (' '.join(subproc_args)))

        sub_ret = subprocess_exe("hadoop", subproc_args, local_log_prefix=subprocess.PIPE)
        if sub_ret['success'] is False:
            if sub_ret['python_exception'] is not None:
                raise sub_ret['python_exception']
            else:
                raise RuntimeError("Unexpected error launching YARN client")

        try:
            job_info = self._parse_java_client_output(sub_ret)
            job_info['stdout'] = sub_ret['stdout']
            job_info['stderr'] = sub_ret['stderr']
            job_info['proc_object'] = sub_ret['proc_object']
            job_info['script_file'] = script_file
        except RuntimeError as e:
            raise RuntimeError("Failed while submitting hadoop job through "\
            "YARN client. Yarn client log:\n" + str(e))

        logger.info('Yarn Application Id: %s' % job_info['app_id'])

        # Todo return job handle
        return job_info

    def _parse_java_client_output(self, subproc_ret):
        """
        Parses output until an application_id is found and returns the info
        found. The process is intended to still be running and printing data
        when this function returns.
        """
        if subproc_ret['return_code'] is not None:
            raise RuntimeError("Must not launch YARN application with blocking client.")
        extracted_info = {}
        app_id_regex = re.compile('.*Submitted application (?P<app_id>application_\w+)$')
        fatal_regex = re.compile('.*FATAL distributedshell.Client:.*')
        full_log = ""
        line = subproc_ret['stderr'].readline()
        fatal_condition_found = False
        while line != '':
            full_log += line
            app_match = re.search(app_id_regex, line)
            if app_match:
                extracted_info['app_id'] = app_match.group('app_id')
                break
            if re.search(fatal_regex, line):
                fatal_condition_found = True

            line = subproc_ret['stderr'].readline()

        if fatal_condition_found or not app_match:
            raise RuntimeError(full_log)
        return extracted_info
