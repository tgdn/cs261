#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Collects data for testing

import sys
import socket
import sqlite3 as s
import argparse

from purple.finance import Trade

# Main table
TABLE = '''
DROP TABLE IF EXISTS trades;
CREATE TABLE trades(
    price NUMERIC,
    size NUMERIC,
    bid NUMERIC,
    ask NUMERIC,
    time TEXT,
    buyer TEXT,
    seller TEXT,
    symbol TEXT,
    sector TEXT
);'''

# connect to sqlite
conn = s.connect('data.db')

def create_tables(cur):
    try:
        cur.executescript(TABLE)
        conn.commit()
    except s.Error, e:
        if conn:
            conn.rollback()
        print "Error: %s" % e.args[0]
        sys.exit(1)

def insert_trade(cur, t):
    global conn
    cur.execute((
        'INSERT INTO '
        'trades (price, size, bid, ask, time, buyer, seller, symbol, sector) '
        'VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);'
    ), (
        t.price, t.size, t.bid, t.ask,
        t.time, t.seller, t.buyer,
        t.symbol, t.sector
    ))
    if conn:
        conn.commit()


def main():
    global conn
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()

    with conn:
        # get db cursor
        cur = conn.cursor()
        # create tables if required
        if (args.init):
            create_tables(cur)

        # Open socket with given paramaters
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('cs261.dcs.warwick.ac.uk', 80))
        except socket.error, e:
            # Oopsy, couldn't connect
            print e
            sys.exit(1)

        line = ''
        firstline = True

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
                        insert_trade(cur, t)
                line = ''
            else:
                line = line + char


if __name__ == '__main__':
    main()
