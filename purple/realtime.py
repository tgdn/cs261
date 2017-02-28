# -*- coding: utf-8 -*-

import os
import sys
import pytz
from datetime import datetime
import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

from purple.db import get_reql_connection
tz = pytz.timezone('Europe/London')

class NotificationManager:
    def add(self, **kwargs):
        with get_reql_connection(db=True) as conn:
            r.table('notifications').insert([
                kwargs
            ]).run(conn, durability='soft')


class AlertManager:
    def add(self):
        with get_reql_connection(db=True) as conn:
            pass


class TaskManager:
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
    def end(pk):
        with get_reql_connection(db=True) as conn:
            r.table('tasks').get(pk).update({
                'ended_at': tz.localize(datetime.now()),
                'terminated': True
            }).run(conn)
