# -*- coding: utf-8 -*-

import os
import sys
import fcntl
import socket
import threading
from contextlib import contextmanager

import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

from purple import db
from purple.finance import Trade

RDB_HOST = 'localhost'
RDB_PORT = '28015'
PURPLE_DB = 'purple'

# write to screen
def stdout_write(s):
    sys.stdout.write(s)
    sys.stdout.flush()

# reset line to write on same line again
def reset_line():
    sys.stdout.write('\r')
    sys.stdout.flush()


class App:
    def __init__(self, args):
        '''
        First, parse command line arguments and decide what to do:
        --init-db                  -> initialise db (tables etc)
        --reset-db                 -> delete tables and data
        -f trades.csv              -> import trades from file
        -s cs261.warw.ac.uk -p 80  -> import trades from live stream
        '''

        if args.reset_db:
            self.reset_rdb()
            self.reset_psql()
            print 'Database dropped successfully.'
        if args.init_db:
            self.init_rdb()
            self.init_psql()
        if args.file:
            self.from_file(args.file)
        if args.stream_url:
            port = args.port or 80
            self.from_stream(url=args.stream_url, port=port)

    def from_file(self, f):
        '''
        Read file containing trading data.
        Read each line and insert in DB.
        f is file handle opened by argparse.

        Trades are commited once the whole file
        has been read. Cancelling command will
        leave database unchanged.
        '''
        # set non blocking i/o
        fd = f.fileno()
        flag = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFL, flag |  os.O_NONBLOCK)
        next(f) # skip header row

        # hold a trade accumulator and a trade count
        tradeacc = 0
        tradecount = 0
        trades_objs = []

        # read line by line
        for line in f:
            # continue if row is parsed correctly
            t = Trade(line)
            if not t.parse_err:
                # get symbol
                symbol = db.SymbolModel.get_or_create(t.symbol)

                # create query
                trade = db.TradeModel(price=t.price, size=t.size, symbol=symbol)
                trades_objs.append(trade)
                tradeacc = tradeacc + 1

                # flush database every 2500 objects
                if tradeacc == 2500:
                    # bulk save for improved performance
                    db.session.bulk_save_objects(trades_objs)
                    db.session.flush()
                    trades_objs = []
                    tradeacc = 0

            # inform user
            tradecount = tradecount + 1
            stdout_write('Trades: {} (Ctrl-C to stop)'.format(tradecount))
            reset_line()

        # save last items out of the loop and commit changes to db
        db.session.bulk_save_objects(trades_objs)
        db.session.commit()

        try:
            f.close()
        except:
            pass

    def from_stream(self, url, port=80):
        '''
        Read live stream of trading data
        and insert into DB.

        Unlike from_file, trades are
        commited every 100 trades
        for better live statistics.
        '''
        # Open socket with given paramaters
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((url, port))

        # store accumulator and trade count
        firstline = True
        tradeacc = 0
        tradecount = 0
        trades_objs = []
        line = ''

        # read character by character until new line '\n'
        # is found. Parse line at that time and continue.
        while 1:
            # read character
            char = sock.recv(1)
            if char == '\n':
                if firstline:
                    firstline = False
                else:
                    t = Trade(line)
                    if not t.parse_err:
                        symbol = db.SymbolModel.get_or_create(t.symbol)
                        trade = db.TradeModel(price=t.price, size=t.size, symbol=symbol)
                        trades_objs.append(trade)
                        tradeacc = tradeacc + 1
                        if tradeacc == 50:
                            # bulk save and commit
                            db.session.bulk_save_objects(trades_objs)
                            db.session.commit()
                            trades_objs = []
                            tradeacc = 0

                    # inform user
                    tradecount = tradecount + 1
                    stdout_write('Trades: {} (Ctrl-C to stop)'.format(tradecount))
                    reset_line()
                line = ''
            else:
                line = line + char

    def reset_rdb(self):
        # connect to rethinkdb and drop database
        with self.get_reql_connection() as conn:
            try:
                r.db_drop(PURPLE_DB).run(conn)
            except RqlRuntimeError:
                pass

    def init_rdb(self):
        # connect to rethinkdb and create database and tables
        with self.get_reql_connection() as conn:
            try:
                r.db_create(PURPLE_DB).run(conn)
                # r.db(PURPLE_DB).table_create('trades').run(conn)
                ## TODO: create alert table (think of design)
            except RqlRuntimeError:
                # fail silently
                # Remember to reset db first to migrate db
                pass
            finally:
                print 'Rethinkdb setup complete.'

    def reset_psql(self):
        db.drop_tables()

    def init_psql(self):
        db.create_tables()

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
