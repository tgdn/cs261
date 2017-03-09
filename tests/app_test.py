# -*- coding: utf-8 -*-

from purple.finance import Trade
from purple.analysis import TradesAnalyser
import pytest
from purple import App
import argparse

parser = argparse.ArgumentParser(description='Purple trading backend')
group = parser.add_mutually_exclusive_group()

parser.add_argument(
    '--init-db', action='store_true',
    help='Create database, tables and indexes.\
    Will have no effect if the db already exists.'
)
parser.add_argument(
    '--reset-db', action='store_true',
    help='Delete database. WARNING: This will delete everything!'
)
group.add_argument(
    '-f', '--file', type=argparse.FileType('r'),
    help='File input to read trade data from.'
)
group.add_argument(
    '-s', '--stream-url', type=str,
    help='URL to a stream of data.'
)
parser.add_argument(
    '-p', '--port', type=int, default=80,
    help='Specify a port to given stream url. (default: 80)'
)

args = parser.parse_args()

app = App(args)
test_file = open('tests/test_csv.csv', 'r')

#We're given an Index Error if our file has been correctly parsed
def test_read_from_file():
	try:
		app.from_file(test_file)
		assert False
	except IndexError:
		assert True