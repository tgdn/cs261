# -*- coding: utf-8 -*-

import os
import sys
import pytz
from datetime import datetime
import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

from purple.db import get_reql_connection

# Set our timezone
tz = pytz.timezone('Europe/London')

# Manage notifications for frontend
class NotificationManager:
    def add(self, **kwargs):
        with get_reql_connection(db=True) as conn:
            r.table('notifications').insert([
                kwargs
            ]).run(conn, durability='soft')

# Manage alerts for frontend
class AlertManager:
    def add(self):
        with get_reql_connection(db=True) as conn:
            pass

# Manage tasks for frontend
class TaskManager:
    # Store a task
    @staticmethod
    def store(**kwargs):
        params = kwargs.copy()
        params['pid'] = os.getpid()
        params['created_at'] = tz.localize(datetime.now())
        params['terminated'] = False

        with get_reql_connection(db=True) as conn:
            res = r.table('tasks').insert(params).run(conn)
            return res['generated_keys'][0]

    @staticmethod
    # End a task
    def end(pk):
        with get_reql_connection(db=True) as conn:
            r.table('tasks').get(pk).update({
                'ended_at': tz.localize(datetime.now()),
                'terminated': True
            }).run(conn)
