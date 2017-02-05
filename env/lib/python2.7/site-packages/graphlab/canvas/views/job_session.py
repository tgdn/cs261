import graphlab.canvas
import graphlab.canvas.views as views
import graphlab.canvas.views.base
import graphlab.canvas.views.job
import graphlab.deploy

import re

import sys
if sys.version_info.major > 2:
    from urllib.parse import unquote
else:
    from urllib2 import unquote

class ScopedSessionView(graphlab.canvas.views.base.BaseView):

    def __init__(self, obj, params=None):
        super(ScopedSessionView, self).__init__(obj, params)
        self.__max_loads = 20 # max number of jobs loaded at each time interval
        self.__child_views = {}
        self.__job_list = {}
        self.objectType = 'Jobs' # override default object type
        self.register_handler('get', 'load_job', self.__load_job_handler)
        self.register_handler('get', 'list_jobs', self.__list_jobs_handler)

    def get_metadata(self):
        data = {
            'sessionname' : 'Dashboard',
        }
        return data

    def __list_jobs_handler(self, url, handler):
        jobname_list = self.__update_child_views()
        handler.write({'jobs': jobname_list})

    def __update_child_views(self):
        # add child views
        if 'Name' in self.obj.list().column_names():
            joblist = list(self.obj.list()['Name'])
            for jobname in joblist:
                if not jobname in self.__child_views:
                    self.__child_views[jobname] = graphlab.canvas.views.job.JobView(jobname)

            return joblist
        else:
            return []

    def __load_job_handler(self, url, handler):
        m = re.match('load_job/(.*)', url)
        jobname = unquote(m.group(1).encode('utf-8').decode()).encode('utf-8').decode()
        if jobname in self.__job_list and self.__job_list[jobname] is not None:
            job = self.__job_list[jobname]
        else:
            job = graphlab.deploy.jobs[jobname]
            self.__job_list[jobname] = job
            # link this loaded job to the child view, so job loading is only needed once
            self.__child_views[jobname]._set_job(job)

        if not job:
            handler.write({"job":None})
            return

        # TODO: need to clean up get job type and get status, same code in job
        jobtype = type(job).__name__
        job_status = job.get_status(_silent=True)

        job_info = {}
        job_info['name'] = job.name
        job_info['status'] = job_status
        job_info['starttime'] = str(job.get_start_time())
        job_info['type'] = jobtype
        job_info['filename'] = jobname # add file name
        # get rest of the completed info
        if job.get_status(_silent=True) == 'Completed':
            job_info['endtime'] = str(job.get_end_time())

        handler.write({"job": job_info})

    def child_views(self):
        self.__update_child_views()
        return self.__child_views

    def get_js_file(self):
        return 'job_session'

    def get_temporary_name(self):
        return ('Jobs',)
