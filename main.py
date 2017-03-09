#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Used to parse command line arguments
import argparse

from purple import App

def main():
    parser = argparse.ArgumentParser(description='Purple trading backend')
    group = parser.add_mutually_exclusive_group()

    # When we want to initialise our db
    parser.add_argument(
        '--init-db', action='store_true',
        help='Create database, tables and indexes.\
        Will have no effect if the db already exists.'
    )
    # When we want to reset our db
    parser.add_argument(
        '--reset-db', action='store_true',
        help='Delete database. WARNING: This will delete everything!'
    )
    # We want to analyse from a file
    group.add_argument(
        '-f', '--file', type=argparse.FileType('r'),
        help='File input to read trade data from.'
    )
    # We want to analyse from a stream
    group.add_argument(
        '-s', '--stream-url', type=str,
        help='URL to a stream of data.'
    )
    # Port of the stream, defaults to 80
    parser.add_argument(
        '-p', '--port', type=int, default=80,
        help='Specify a port to given stream url. (default: 80)'
    )

    args = parser.parse_args()

    # Run our app with arguments
    App(args)

if __name__ == '__main__':
    main()
