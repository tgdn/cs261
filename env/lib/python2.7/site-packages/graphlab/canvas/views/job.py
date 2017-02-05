import graphlab.canvas
import graphlab.canvas.views.base
import graphlab.canvas.views.task
import graphlab.deploy
from graphlab.util import _is_string, file_util as _file_util

import os as _os

import sys
if sys.version_info.major > 2:
    long = int


class JobView(graphlab.canvas.views.base.BaseView):

    def __init__(self, obj, params=None):
        super(JobView, self).__init__(obj, params)
        self.__child_views = {}
        self.__job = None
        self.objectType = 'Job'
        self.register_handler('get', 'log_file_path_list', self.__get_log_file_path_list)
        self.register_handler('get', 'get_log_file_content', self.__get_log_file_content)
        self.register_handler('get', 'load_job_info', self.__load_job_info)
        self.register_handler('get', 'get_task_status', self.__get_task_status_handler)

    def _set_job(self, job=None):
        if job is not None:
            self.__job = job

    def __load_job(self):
        if not self.__job:
            self.__job = graphlab.deploy.jobs[self.obj]
        return self.__job

    def get_metadata(self):
        job = self.__load_job()
        return {
            'jobname': [job.name],
        }

    # recursively parses the job parameters and stringify to a dictionary of strings
    def __convert_params_to_json(self, params, recursive_level=2):
        table = None
        # task.get_inputs() would always returns a dictionary; however, we need to take
        # care of the recursive case when we have list/dict within the top-level dict
        if isinstance(params, dict):
            table = dict()
            for k in params.keys():
                v = params[k]
                if isinstance(v, (int, float, bool)) or _is_string(v):
                    table[k] = v
                elif isinstance(v, (dict, list, tuple, set)):
                    if recursive_level > 0:
                        table[k] = self.__convert_params_to_json(v, recursive_level-1)
                    else:
                        table[k] = '...'
                else:
                    table[k] = '...'
        else:
            table = []
            for v in params:
                if isinstance(v, (int, float, bool)) or _is_string(v):
                    table.append(v)
                elif isinstance(v, (dict, list, tuple, set)):
                    if recursive_level > 0:
                        table.append(self.__convert_params_to_json(v, recursive_level-1))
                    else:
                        table.append('...')
                else:
                    table.append('...')
        return table

    # pretty prints the parameters, assuming we only get the following types in
    # values: int, float, bool, basestring, array, dict.
    def __pretty(self, params):
        out = ""
        for k in params.keys():
            out += k + ": "
            out += str(params[k]) + "\n"
        return out

    def __load_job_info(self, url, handler):
        job = self.__load_job()
        code = job.get_code()
        tasks = []
        parameters = job.get_parameters()
        # convert parameters to json format
        for key in parameters:
            task = dict()
            task['name'] = key
            params = self.__convert_params_to_json(parameters[key])
            task['inputs'] = self.__pretty(params)
            tasks.append(task)

        # ec2 commander dns name
        dns = job.ec2_dns_name if hasattr(job, 'ec2_dns_name') else ''

        jobtype = type(job).__name__
        job_status = job.get_status(_silent=True)

        # get env info
        env = {"name" : job.environment.name, "type": type(job.environment).__name__}

        # get start and end time
        start_time = str(job.get_start_time())
        end_time = str(job.get_end_time())

        handler.write({
            'jobname' : [job.name],
            'packages': list(job.get_required_packages()),
            'code' : code,
            'tasks' : tasks,
            'env' : env,
            'status' : job_status,
            'jobtype' : jobtype,
            'starttime' : start_time,
            'endtime' : end_time,
            'dns' : dns
        })

    def __get_task_status_handler(self, url, handler):
        job = self.__load_job()
        task_status = {}
        metrics = job.get_metrics()
        if metrics:
            for m in metrics:
                task_status[m['task_name']] = m
        handler.write({'task_status': task_status})

    def get_staticdata(self):
        return self.get_metadata()

    def __get_log_file_content(self, url, handler):
        """
        Get and return the log file content
        """
        log_file_path = str(handler.get_argument("log", None))
        job = self.__load_job()
        content = ""
        max_size = long(1048576) # max size is 1mb
        status_code = 200
        if log_file_path:
            try:
                if _file_util.is_local_path(log_file_path):
                    if _os.path.getsize(log_file_path) > max_size:
                        raise RuntimeError("Cannot read file greater than max size.")
                    else:
                        content = self.__load_local_log_file(log_file_path)
                elif _file_util.is_s3_path(log_file_path):
                    content = _file_util.read_file_to_string_s3(log_file_path,
                                                        max_size,
                                                        job.environment.get_credentials())
                elif _file_util.is_hdfs_path(log_file_path):
                    content = _file_util.read_file_to_string_hdfs(log_file_path,
                                                        max_size,
                                                        job.environment.hadoop_conf_dir)
                else:
                    status_code = 404
                    content = "Log file path (%s) is not valid." % log_file_path
            except RuntimeError:
                status_code = 413
                content = "File size too large. Please load log file manually at %s." % log_file_path

        handler.set_status(status_code)
        handler.set_header("Content-Type", "text/plain")
        handler.write(content)

    def __get_log_file_path_list(self, url, handler):
        """
        Returns a list of log file path for this job
        """
        job = self.__load_job()

        # get the directory that contains all the logs
        log_file_path = job.get_log_file_path()
        path_list = []

        # list the directory to get the full path to each log
        if _file_util.is_s3_path(log_file_path):
            ec2_log_list = _file_util.list_s3(log_file_path,
                                              job.environment.get_credentials())
            if ec2_log_list and len(ec2_log_list) > 0:
                path_list.extend([log['path'] for log in ec2_log_list])
        elif _file_util.is_hdfs_path(log_file_path):
            hdfs_log_list = _file_util.list_hdfs(log_file_path,
                                                   job.environment.hadoop_conf_dir)
            if hdfs_log_list and len(hdfs_log_list) > 0:
                path_list.extend([log['path'] for log in hdfs_log_list])
        else:
            path_list.append(log_file_path)

        handler.write({'log_file_list': path_list})

    def __load_local_log_file(self, log_file_path):
        data = ""
        try:
            with open(log_file_path, 'r') as f:
                data = f.read()
            f.close()
        except IOError:
            data = "Unable to read file from %s " % log_file_path
        return data

    def child_views(self):
        return self.__child_views

    def get_js_file(self):
        return 'job'
