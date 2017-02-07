# -*- coding: utf-8 -*-

import os
import sys
import fcntl

import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

from purple.finance import Trade

RDB_HOST = 'localhost'
RDB_PORT = '28015'
PURPLE_DB = 'purple'

class App:
    def __init__(self, argv):
        self.argv = argv
        self.setup_db()
        self.connect_db()

        ##
        ## TODO: parse args so we can decide whether to use file or url
        ##

        if (len(argv) == 2):
            filename = argv[1]

            with open(filename, 'r') as f:
                # set non blocking i/o
                fd = f.fileno()
                flag = fcntl.fcntl(fd, fcntl.F_GETFD)
                fcntl.fcntl(fd, fcntl.F_SETFL, flag |  os.O_NONBLOCK)
                lineno = 0

                for line in f:
                    if lineno == 0:
                        lineno = 1
                        continue

                    t = Trade(line).save(self.rdb_conn)
                    print '{} {} {}'.format(t.currency, t.price, t.sector)
        else:
            sys.stderr.write('Specify a filename: python main.py [filename]')

        # finally
        self.close_db()

    def setup_db(self):
        connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        try:
            r.db_create(PURPLE_DB).run(connection)
            r.db(PURPLE_DB).table_create('trades').run(connection)
            r.db(PURPLE_DB).table_create('sectors', primary_key='name').run(connection)
            r.db(PURPLE_DB).table_create('symbols', primary_key='name').run(connection)
        except RqlRuntimeError:
            # fail silently
            # Remember to destroy if you want this to be successful
            pass
        finally:
            connection.close()
        print 'Database setup complete.'

    def connect_db(self):
        try:
            self.rdb_conn = r.connect(host=RDB_HOST, port=RDB_PORT, db=PURPLE_DB)
        except RqlDriverError:
            sys.stderr.write('No db connection could be established.')
            sys.exit(1)

    def close_db(self):
        try:
            this.rdb_conn.close()
        except:
            pass
