# -*- coding: utf-8 -*-

from datetime import datetime

class Trade:
    def __init__(self, row):
        split_row = row.split(',')
        # datetime is given to us such as this one
        #'2017-01-13 15:26:41.917266'

        self.time = datetime.strptime(split_row[0], '%Y-%m-%d %H:%M:%S.%f')
        self.buyer = split_row[1]
        self.seller = split_row[2]
        self.price = float(split_row[3])
        self.size = int(split_row[4])
        self.currency = split_row[5]
        self.symbol = split_row[6]
        self.sector = split_row[7]
        self.bid = float(split_row[8])
        self.ask = float(split_row[9])


##
## TODO: add more finance related classes to do work
##
