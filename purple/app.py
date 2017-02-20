# -*- coding: utf-8 -*-

import os
import fcntl
import socket

from purple import db
from purple.finance import Trade
from purple.analysis import TradesAnalyser


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

        #
        # Tested this with less trades to store before commiting
        # (tradeacc_limit) but havent found a big difference in
        # the time it takes.
        #
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
        commited every 50 trades
        for better live statistics.
        '''
        # Open socket with given paramaters
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((url, port))

        line = ''
        firstline = True
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
