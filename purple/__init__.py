# -*- coding: utf-8 -*-

import os
import sys
import pytz
import fcntl
import socket
#import threading
from datetime import datetime

import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

from purple import db
from purple.anomalous_trade_finder import AnomalousTradeFinder
from purple.finance import Trade
from purple.analysis import TradesAnalyser

tz = pytz.timezone('Europe/London')

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
            db.drop_tables()
            print 'Database dropped successfully.'
        if args.init_db:
            db.create_tables()
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
        # tradeacc = 0
        # tradecount = 0
        # trades_objs = []

        # anomaly_identifier = AnomalousTradeFinder()

        trades_analyser = TradesAnalyser(tradeacc_limit=1000)

        # read line by line
        for line in f:
            # continue if row is parsed correctly
            t = Trade(line)
            if not t.parse_err:
                trades_analyser.add(t)
        trades_analyser.force_commit()

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

        trades_analyser = TradesAnalyser(tradeacc_limit=50)

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
                        trades_analyser.add(t, commit=True)
                line = ''
            else:
                line = line + char

    def flag(self, trade):
        trade.flag(True)
        with db.get_reql_connection(db=True) as conn:
            r.table('alerts').insert([{
                    'time': tz.localize(datetime.now()),
                    'trade_pk': trade.id,
            }]).run(conn, durability='soft')
