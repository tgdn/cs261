"""
GraphLab Create can launch an EC2 instance with one line of Python code, then
all GraphLab Create operations are executed on that instance.

Explore the functionalities in detail through the API documentation, or learn
more from the `Running in the Cloud Tutorial. <https://turi.com/learn/notebooks/running_in_the_cloud.html>`_

"""

'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''

from ._ec2 import get_credentials, list_instances, set_credentials
