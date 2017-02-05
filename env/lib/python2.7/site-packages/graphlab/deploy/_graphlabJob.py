import sys as _sys
import os as _os

sys_path = _os.environ.get('__GL_SYS_PATH__')
if sys_path is not None:
    print('Setting system path to: %s' % sys_path)
    _sys.path = sys_path.split(_os.path.pathsep)

import time
import json
import logging as _logging
import traceback
from graphlab.deploy._job import Job
from graphlab.deploy._executionenvironment import LocalExecutionEnvironment
def write_job_info(path, value):
    try:
        with open(path, 'w') as f:
            json.dump(value, f)
    except Exception as e:
        _logging.info('Exception trying to write job info')
        _logging.info(e)
        pass

if __name__ == '__main__':

    if(len(_sys.argv) != 2):
        print('Did not get required args. Usage:')
        print("%s <job file> " % _os.path.basename(_sys.argv[0]))
        exit(1)

    print('\nargs received: %s' % _sys.argv)
    serialized_job_location = _sys.argv[1]
    job_work_dir = _os.path.split(serialized_job_location)[0]
    print('job_work_dir: %s ' % job_work_dir)


    status_path = _os.path.join(job_work_dir, 'status')
    metrics_path = _os.path.join(job_work_dir, 'metrics')
    start_time = int(time.time())
    try:
        job = Job._deserialize(serialized_job_location)

        # Now use the local launcher to run the job in this thread, will be
        # async w.r.t other thread.
        write_job_info(status_path, {'status': 'Running', 'start_time':start_time, 'end_time':None})

        job._metrics = []
        for cur_step in job._stages:
            for task in cur_step:
                print('running task %s' % task.name)
                task_metrics = LocalExecutionEnvironment._run_task(task, job, job._task_output_paths[task])
                job._metrics.append(task_metrics)
                write_job_info(metrics_path, job._metrics)

        # write out job status
        end_time = int(time.time())
        write_job_info(status_path, {'status': 'Completed', 'start_time':start_time, 'end_time':end_time})

        exit(0)

    except Exception as e:
        trace = traceback.format_exc()
        err_msg = "Job execution failed.\n"
        err_msg += "Traceback (most recent call last)\n %s\n" % trace
        err_msg += "Error type    : %s\n" % e.__class__.__name__
        err_msg += "Error message : %s\n" % str(e)

        print(err_msg)

        end_time = int(time.time())
        write_job_info(status_path, {'status':'Failed', 'error': err_msg, 'start_time':start_time, 'end_time':end_time})

        exit(1)
