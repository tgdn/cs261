# -*- coding: utf-8 -*-

import sys
import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

from purple.db import get_reql_connection

class NotificationManager:

    def add(self, **kwargs):
        with get_reql_connection(db=True) as conn:
            r.table('notifications').insert([
                kwargs
            ]).run(conn, durability='soft')
