# -*- coding: utf-8 -*-

############################
# Class that holds a trade #
############################

import pytz
from datetime import datetime

import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

# Set our timezone
tz = pytz.timezone('Europe/London')

class Trade:
    def __init__(self, row):
        self.parse_err = False

        try:
            # Try splitting
            split_row = row.split(',')

            # Localize timezone
            self.time = tz.localize(datetime.strptime(split_row[0], '%Y-%m-%d %H:%M:%S.%f'))
            self.buyer = split_row[1]
            self.seller = split_row[2]
            self.price = float(split_row[3])
            self.size = int(split_row[4])
            self.currency = split_row[5]
            self.symbol = split_row[6]
            self.sector = split_row[7]
            self.bid = float(split_row[8])
            self.ask = float(split_row[9])
        except:
            self.parse_err = True
