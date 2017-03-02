# -*- coding: utf-8 -*-

import os
import pytz
import fcntl
import signal
import socket
import hashlib
import atexit
from datetime import datetime
from threading import Timer
import rethinkdb as r

from purple import db
from purple.realtime import NotificationManager, TaskManager
from purple.finance import Trade
from purple.analysis import TradesAnalyser

tz = pytz.timezone('Europe/London')

# process globals
TASK_PK = None
TASK_ENDED = False
notification_manager = NotificationManager()
task_manager = TaskManager()

def before_exit():
    '''
    Will store the tasks exit in rethinkdb
    '''
    global TASK_PK
    global TASK_ENDED

    if TASK_PK and not TASK_ENDED:
        task_manager.end(TASK_PK)
        # print '\nNow notifying'
        notification_manager.add(
            level = 'info',
            message = 'Background task ended',
            datetime = tz.localize(datetime.now())
        )

# register exit handlers
atexit.register(before_exit)
signal.signal(signal.SIGTERM, before_exit)


class App:
    def __init__(self, args):
        '''
        First, parse command line arguments and decide what to do:
        --init-db                  -> initialise db (tables etc)
        --reset-db                 -> delete tables and data
        -f trades.csv              -> import trades from file
        -s cs261.warw.ac.uk -p 80  -> import trades from live stream
        '''
        global TASK_ENDED
        global TASK_PK

        if args.reset_db:
            db.drop_tables()
        if args.init_db:
            db.create_tables()

        # check whether there is no analysis happening currently
        if args.file or args.stream_url:
            with db.get_reql_connection(db=True) as conn:
                task_count = r.table('tasks').filter(r.row['terminated'] == False).count().run(conn)
                if task_count:
                    notification_manager.add(
                        level = 'warning',
                        title = 'Cannot launch task',
                        message = 'End current task before you can start new analysis',
                        datetime = tz.localize(datetime.now())
                    )
                    return

        if args.file:
            TASK_PK = task_manager.store(task='analysis', type='file')
            self.from_file(args.file)
        if args.stream_url:
            port = args.port or 80
            TASK_PK = task_manager.store(task='analysis', type='stream')
            self.from_stream(url=args.stream_url, port=port)

        # Task will be ended before_exit

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

        #The blocksize we are using reading into our hash buffer
        BLOCKSIZE = 65536

        #Take a SHA1 hash of our file
        file_buff = f.read(BLOCKSIZE)

        while len(file_buff) > 0:
            hashlib.sha1().update(file_buff)
            file_buff = f.read(BLOCKSIZE)

        sha1_hash = hashlib.sha1().hexdigest()

        #Return to beginning of file (after header)
        f.seek(1)

        # Tested this with less trades to store before commiting
        # (tradeacc_limit) but havent found a big difference in
        # the time it takes.

        trades_analyser = TradesAnalyser(tradeacc_limit=1000)
        # read line by line
        for line in f:
            # continue if row is parsed correctly
            t = Trade(line)
            if not t.parse_err:
                trades_analyser.add(t,sha1_hash)
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
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((url, port))
        except socket.error, e:
            # Oopsy, couldn't connect
            print e
            notification_manager.add(
                level = 'error',
                title = 'Connection not established',
                message = (
                    'There was an issue while trying to connect to the host: {}'.format(str(e))
                ),
                datetime = tz.localize(datetime.now())
            )
            return

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
                        trades_analyser.add(t, None, commit=True)
                line = ''
            else:
                line = line + char
