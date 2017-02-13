# -*- coding: utf-8 -*-

import os
import sys
import fcntl
import socket
from contextlib import contextmanager

import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

from purple.finance import Trade

RDB_HOST = 'localhost'
RDB_PORT = '28015'
PURPLE_DB = 'purple'

# write to stdout
def stdout_write(s):
    sys.stdout.write(s)
    sys.stdout.flush()

# reset line to write on same line again
def reset_line():
    sys.stdout.write('\r')
    sys.stdout.flush()


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

    def from_file(self, f):
        '''
        Read file containing trading data.
        Read each line and insert in DB.
        File is opened by argparse.
        '''
        # set non blocking i/o
        fd = f.fileno()
        flag = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFL, flag |  os.O_NONBLOCK)
        next(f) # skip header row

        tradecount = 0
        with self.get_reql_connection(True) as conn:
            for line in f:
                # insert
                Trade(line).save(conn)

                # inform user
                tradecount = tradecount + 1
                stdout_write('Trades: {} (Ctrl-C to stop)'.format(tradecount))
                reset_line()
        try:
            f.close()
        except:
            pass

    def from_stream(self, url, port=80):
        '''
        Read live stream of trading data
        and insert into DB.
        '''
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((url, port))

        firstline = True
        tradecount = 0
        line = ''
        with self.get_reql_connection(True) as conn:
            while 1:
                # read character
                char = sock.recv(1)
                if char == '\n':
                    if firstline:
                        firstline = False
                    else:
                        # insert
                        Trade(line).save(conn)

                        # inform user
                        tradecount = tradecount + 1
                        stdout_write('Trades: {} (Ctrl-C to stop)'.format(tradecount))
                        reset_line()
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
    def get_reql_connection(self, db=False):
        """
        Make rdb connection available as context manager generator.
        ie:
        with self.get_reql_connection(db=True) as conn:
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
