# -*- coding: utf-8 -*-

# Used for file management
import os
# Used for timezone handling
import pytz
import sys
# Used for file handling
import fcntl
# Used for exit handling
import signal
# Used for connection to feed
import socket
# Used to generate hash of csv
import hashlib
# Used for exit handling
import atexit
# Used for time handling
import time
# Used for datetime handling
from datetime import datetime, timedelta
# rethinkDB
import rethinkdb as r

# PostgreSQL
from purple import db
from purple.realtime import NotificationManager, TaskManager
from purple.finance import Trade
from purple.analysis import TradesAnalyser

# Set our timezone
tz = pytz.timezone('Europe/London')

# process globals
TASK_PK = None
TASK_ENDED = False
FILE_HANDLE = None
notification_manager = NotificationManager()
task_manager = TaskManager()

# Handles process ending
def before_exit(signum=None, frame=None):
    '''
    Will store the tasks exit in rethinkdb
    '''
    global TASK_PK, TASK_ENDED, FILE_HANDLE

     # close file
    if FILE_HANDLE:
        try:
            f.close()
        except:
            pass

    if TASK_PK and not TASK_ENDED:
        task_manager.end(TASK_PK)
        notification_manager.add(
            level = 'info',
            message = 'Background task ended',
            datetime = tz.localize(datetime.now())
        )

    # Do quit
    if signum:
        sys.exit(0)

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
        -s cs261.dcs.warwick.ac.uk -p 80  -> import trades from live stream
        '''
        global TASK_ENDED
        global TASK_PK

        # Drop or initialise the PostgreSQL db as necessary
        if args.reset_db:
            db.drop_tables()
        if args.init_db:
            db.create_tables()

        # Check whether there is no analysis happening currently
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

        # Analyse a file
        if args.file:
            TASK_PK = task_manager.store(task='analysis', type='file')
            self.from_file(args.file)
        # Analyse a stream
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

        Trades are commited every `tradeacc_limit`
        but analysis is only performed at the end.
        Cancelling command will store some trades
        in DB but wont perform analysis
        '''
        global FILE_HANDLE
        FILE_HANDLE = f

        # Set non blocking i/o
        fd = f.fileno()
        flag = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFL, flag |  os.O_NONBLOCK)

        # The blocksize we are using reading into our hash buffer
        BLOCKSIZE = 65536

        # Take a SHA1 hash of our file
        file_buff = f.read(BLOCKSIZE)

        while len(file_buff) > 0:
            hashlib.sha1().update(file_buff)
            file_buff = f.read(BLOCKSIZE)

        sha1_hash = hashlib.sha1().hexdigest()

        # Return to beginning of file (after header)
        f.seek(1)

        # Tested this with less trades to store before commiting
        # (tradeacc_limit) but havent found a big difference in
        # the time it takes.

        trades_analyser = TradesAnalyser(tradeacc_limit=1000)
        # Read line by line
        print "Adding lines for analysis"
        for line in f:
            # Continue if row is parsed correctly
            t = Trade(line)
            if not t.parse_err:
                trades_analyser.add(t, sha1_hash, True, commit=True)

        trades_analyser.force_commit()
        print "Lines added to memory, beginning anomaly detection"

        # Calculate stats once all trades added
        trades_analyser.alert_stats(True, True)

        # Close the file
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
        firstday = True

        # Open socket with given paramaters
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
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

        # Read character by character until new line '\n'
        # is found. Parse line at that time and continue.
        while 1:
            # Read character
            try:
                char = sock.recv(1)
                if char == '\n':
                    if firstline:
                        firstline = False
                    else:
                        t = Trade(line)
                        if not t.parse_err:
                            # Add trade if it is correct
                            trades_analyser.add(t, None, firstday, commit=True)
                    line = ''
                else:
                    line = line + char
            # The feed is down, we've got to analyse then reconnect
            except socket.timeout:
                print "Connection lost, attempting to reconnect"
                disconnected = True
                # Only analyse if the day of trades is over
                if datetime.now().strftime('%H') == '00':
                    print "Beginning analysis"
                    trades_analyser.alert_stats(firstday, False)
                    firstday = False
                    # Wait for 5 minutes until the feed is accepting connections again
                    time.sleep(300)
                else:
                    # Feed has gone down, notify front end and reconnect
                    print "Feed appears to be down"
                    notification_manager.add(
                        level = 'error',
                        title = 'Feed appears to be down',
                        message = (
                            'The data feed that we are trying to connect to is currently down. Attempting to reconnect.'
                        ),
                        datetime = tz.localize(datetime.now())
                    )

                # Attempt to reconnect
                while disconnected:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        sock.connect((url,port))
                        sock.settimeout(3)
                        disconnected  = False
                        print "Reconnected to the feed!"
                    except:
                        pass
