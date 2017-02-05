"""
GraphLab Create offers multiple ways to work with your data beyond your desktop
or laptop.

- Batch processing with job scheduling in a distributed system, using a
  Hadoop Yarn or EC2 cluster running Turi Distributed.

Explore the detailed functionality via the following API documentation or the
`Deployment chapter of the User Guide
<https://turi.com/learn/userguide/deployment/introduction.html>`_, as well as the
`Gallery <https://turi.com/learn/gallery>`_.

Where did predictive_service go? It is now in its own package, psclient.
"""

__all__ = ['environments', 'jobs', 'job', 'map_job']

# Artifacts
from ._job import Job
from ._task import Task as _Task
from .environment import _Environment

# Sessions
from . import _session
_default_session = _session._open()

from graphlab.util import _is_string

import sys as _sys

environments = _session.ScopedSession(_default_session, _Environment, """
View and manage Environments available in the session. Environments currently
available can be listed, loaded, and deleted using this object.

Examples
--------

.. sourcecode:: python

  >>> my_env = graphlab.deploy.environment.Local("init-ex2-env")
  >>> graphlab.deploy.environments
  Environment(s):
  +-------+--------------+------------------+------------------+
  | Index |     Name     |       Type       | Unsaved changes? |
  +-------+--------------+------------------+------------------+
  |   0   | init-ex2-env | LocalEnvironment |       Yes        |
  +-------+--------------+------------------+------------------+

  # Load an environment by index number.
  >>> environment = graphlab.deploy.environments[0]

  # Load an environment by name.
  >>> environment = graphlab.deploy.environments['init-ex2-env']

  # Delete an environment by name (deleting by index number also supported).
  >>> del graphlab.deploy.environments['init-ex2-env']

""")

jobs = _session.ScopedSession(_default_session, Job, """
View and manage all Jobs in the session. Jobs currently available can be
listed, loaded, deleted, and can be visualized in Canvas using this object.

Examples
--------

.. sourcecode:: python

  # Monitor all jobs
  >>> gl.deploy.jobs
  +-------+-------------+--------------------------+---------------------------+
  | Index | Environment |           Name           |       Creation date       |
  +-------+-------------+--------------------------+---------------------------+
  |   0   |    async    | add-Feb-11-2015-00-39-32 | 2015-02-11 00:39:32+00:00 |
  +-------+-------------+--------------------------+---------------------------+

  # Load a Job by index number:
  >>> job = graphlab.deploy.jobs[0]

  # Load a Job by name:
  >>> job = graphlab.deploy.jobs['job-print-hello-world-task-init-ex3-env-1407900916']

  # Visualize a Job in Canvas.
  >>> job.show()

  # Visualize a list of Jobs in Canvas.
  >>> graphlab.deploy.jobs.show()

  # to delete a Job by index number (deleting by name also supported):
  >>> del graphlab.deploy.jobs[0]
""")

from . import job

from ._ec2_config import Ec2Config
from .ec2_cluster import Ec2Cluster
from .hadoop_cluster import HadoopCluster
from . import ec2_cluster
from . import hadoop_cluster
