# -*- coding: utf-8 -*-

from purple.finance import Trade
from purple.analysis import TradesAnalyser
import pytest


TRADE_ROW = '2017-01-13 15:26:41.917266,w.tuffnell@janestreetcap.com,j.newbury@citadel.com,469.74,15952,GBX,AV.L,Financial,469.08,469.74'
TRADE_ROW1 = '2017-01-13 15:26:51.272423,j.lewis@jlb.com,h.smith@bank.com,473.53,10000,GBX,AV.L,Financial,472.68,473.53'
TRADE_ROW2 = '2017-01-13 15:26:54.258723,m.williams@fake.com,q.fake@fake.biz,474.12,12000,GBX,AV.L,Financial,473.98,474.12'
t = Trade(TRADE_ROW)

def test_add():
	trades_analyser = TradesAnalyser(tradeacc_limit=1000)
	trades_analyser.add(t,"a",True)

	assert trades_analyser.trades_objs[0]["price"] == t.price

######################################################################
#                            Manual Testing                          #
######################################################################

# These manual tests should be carried out everytime any changes are made to the analysis.py file.

# stdout_write and reset_line were tested through inspection (that they produce the correct stdout output)

# force_commit was tested through inspection. Ensure with a basic query that the correct tuples are added to the db.

# save_load was tested through inspection. Ensure with a  basic query that the correct tuples are inserted.

# get_symbol was tested through inspection. Query the db to check that all symbols are correctly inserted.

# flag was tested through inspection. Query the db to check that the correct trade has been flagged after the function is called
# with a custom anomaly.

# alert was tested through inspection, in the same way as flag but with a query in rethinkdb instead. (Check the correct item inserted into alerts)

# For the function alert_stats, each of the individual components was tested. The anomaly detection has been detected through
# unit tests in anomalous_trade_finder_test.py. This function uses the funtions alert, and flag which themselves were tested
# through inspection. All of the components correctly worked together
