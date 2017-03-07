# -*- coding: utf-8 -*-

import pytest
from purple.finance import Trade
from purple.anomalous_trade_finder import AnomalousTradeFinder
from numpy import std, mean
from datetime import datetime

test_finder = AnomalousTradeFinder()
TRADE_ROW = '2017-01-13 15:26:41.917266,w.tuffnell@janestreetcap.com,j.newbury@citadel.com,469.74,15952,GBX,AV.L,Financial,469.08,469.74'
TRADE_ROW1 = '2017-01-13 15:26:51.272423,j.lewis@jlb.com,h.smith@bank.com,473.53,10000,GBX,AV.L,Financial,472.68,473.53'
TRADE_ROW2 = '2017-01-13 15:26:54.258723,m.williams@fake.com,q.fake@fake.biz,474.12,12000,GBX,AV.L,Financial,473.98,474.12'
t = Trade(TRADE_ROW)
t1 = Trade(TRADE_ROW1)
t2 = Trade(TRADE_ROW2)

def test_history_add():
	test_finder = AnomalousTradeFinder()
	test_finder.add(t,1)
	assert len(test_finder.trade_history[t.symbol]) == 1

def test_stats_add():
	test_finder = AnomalousTradeFinder()
	test_finder.add(t,1)
	assert len(test_finder.stats[t.symbol]) == 7

def test_add_correct_delta():
	test_finder = AnomalousTradeFinder()
	test_finder.add(t,1)
	test_finder.add(t1,2)
	assert test_finder.trade_history[t.symbol][-1]["price_delta"] == 3.79

def test_welford():
	test_finder = AnomalousTradeFinder()
	count = 6
	correct_stdev = std([469.74,473.53,474.12, 475.82, 479.05, 482.33])
	current_stdev = std([469.74,473.53,474.12, 475.82, 479.05])
	current_mean = mean([469.74,473.53,474.12, 475.82, 479.05])

	assert round(correct_stdev,3) == round(test_finder.welford(count,current_stdev,current_mean,482.33)["stdev"],3)

def test_calculate_trades_per_min():
	test_finder = AnomalousTradeFinder()
	test_finder.add(t,1)
	test_finder.add(t1,2)
	test_finder.add(t2,3)
	time = datetime.strptime('15:31:21',"%H:%M:%S")
	test_finder._calculate_trades_per_min(time, 3,'AV.L')
	assert test_finder.stats['AV.L']['trade_count_per_min'] == 3

def test_calculate_anomalies_first_day():
	test_finder = AnomalousTradeFinder()
	test_finder.add(t,1)
	test_finder.add(t1,2)
	test_finder.add(t2,3)
	test_finder.calculate_anomalies_first_day(True)

	assert test_finder.stats['AV.L']['delta_mean'] == mean([0,3.79,0.59])
	assert test_finder.stats['AV.L']['delta_stdev'] == std([0,3.79,0.59])
	assert test_finder.stats['AV.L']['vol_mean'] == mean([15952,10000,12000])
	assert test_finder.stats['AV.L']['vol_stdev'] == std([15952,10000,12000])
	assert test_finder.stats['AV.L']['day_price_change_mean'] == 474.12 - 469.74
	assert test_finder.stats['AV.L']['price_change_percentage'] == (474.12 / 473.53)

def test_calculate_fat_finger():
	test_finder = AnomalousTradeFinder()
	test_finder.add(t,1)
	test_finder.add(t1,2)
	test_finder.add(t2,3)
	test_finder.stats['AV.L']['delta_mean'] = 1.46
	test_finder.stats['AV.L']['delta_stdev'] = 0.1
	test_finder.stats['AV.L']['vol_mean'] = mean([15952,10000,12000])
	test_finder.stats['AV.L']['vol_stdev'] = 10

	deltas = [x["price_delta"] for x in test_finder.trade_history['AV.L']]
	ids = [x["id"] for x in test_finder.trade_history['AV.L']]
	times = [x["time"] for x in test_finder.trade_history['AV.L']]
	volumes = [x["volume"] for x in test_finder.trade_history['AV.L']]

	test_finder.calculate_fat_finger(volumes,deltas,ids,times,'AV.L')

	assert len(test_finder.anomalous_trades) == 6

def test_calculate_anomalies_single_trade():
	test_finder = AnomalousTradeFinder()
	test_finder.add(t,1)
	test_finder.add(t1,2)

	test_finder.stats['AV.L']['delta_mean'] = mean([0,3.79])
	test_finder.stats['AV.L']['delta_stdev'] = std([0,3.79])
	test_finder.stats['AV.L']['vol_mean'] = mean([15952,10000])
	test_finder.stats['AV.L']['vol_stdev'] = std([15952,10000])
	test_finder.stats['AV.L']['trade_count'] = 2
	test_finder.stats['AV.L']['total_vol_mean'] = -1
	test_finder.stats['AV.L']['day_price_change_mean'] = -1

	test_finder.prev_trades['AV.L'] = 473.53

	test_finder.calculate_anomalies_single_trade(t2,3)
	assert round(test_finder.stats['AV.L']['delta_mean'],2) == round(mean([0,3.79,0.59]),3)
	assert round(test_finder.stats['AV.L']['delta_stdev'],3) == round(std([0,3.79,0.59]),3)
	assert round(test_finder.stats['AV.L']['vol_mean'],3) == round(mean([15952,10000,12000]),3)
	assert round(test_finder.stats['AV.L']['vol_stdev'],3) == round(std([15952,10000,12000]),3)




