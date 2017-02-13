# -*- coding: utf-8 -*-

import os
import sys
import fcntl
import socket
import requests
from contextlib import contextmanager, closing

import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

from purple.finance import Trade

RDB_HOST = 'localhost'
RDB_PORT = '28015'
PURPLE_DB = 'purple'

class App:
    def __init__(self, args):
        # parse cmd line args and execute stuff
        if args.reset_db:
            self.reset_rdb()
        if args.init_db:
            self.init_rdb()
        if args.file:
            self.from_file(args.file)
        if args.stream_url:
            port = args.port or 80
            self.from_stream(url=args.stream_url, port=port)

        print args

    def from_file(self, filename):
        with open(filename, 'r') as f:
            # set non blocking i/o
            fd = f.fileno()
            flag = fcntl.fcntl(fd, fcntl.F_GETFD)
            fcntl.fcntl(fd, fcntl.F_SETFL, flag |  os.O_NONBLOCK)
            next(f) # skip header row

            with self.get_reql_connection(PURPLE_DB) as conn:
                for line in f:
                    Trade(line).save(conn)

    def from_stream(self, url, port=80):
        # TODO: read from stream
        print 'hello'
        server_name = (url, port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(server_name)

        line = ''
        with self.get_reql_connection() as conn:
            while 1:
                char = sock.recv(1)
                if char == '\n':
                    #print line
                    Trade(line).save(conn)
                    line = ''
                else:
                    line = line + char

    def reset_rdb(self):
        with self.get_reql_connection() as conn:
            try:
                r.db_drop(PURPLE_DB).run(conn)
            except RqlRuntimeError:
                pass
            finally:
                print 'Database dropped successfully.'

    def init_rdb(self):
        with self.get_reql_connection() as conn:
            try:
                r.db_create(PURPLE_DB).run(conn)
                r.db(PURPLE_DB).table_create('trades').run(conn)
                ## TODO: create alert table (think of design)
            except RqlRuntimeError:
                # fail silently
                # Remember to reset db first to migrate db
                pass
            finally:
                print 'Database setup complete.'

    @contextmanager
    def get_reql_connection(db=None):
        """
        Make rdb connection available as context manager generator.
        ie:
        with self.get_reql_connection('somedb') as conn:
            r.table('sometable').run(conn)
        """
        try:
            if db:
                rdb_conn = r.connect(host=RDB_HOST, port=RDB_PORT, db=PURPLE_DB)
            else:
                rdb_conn = r.connect(host=RDB_HOST, port=RDB_PORT)
        except RqlDriverError:
            sys.stderr.write('Rethinkdb: No db connection could be established.')
            sys.exit(1)

        try:
            yield rdb_conn
        finally:
            rdb_conn.close()
