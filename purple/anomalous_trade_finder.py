# -*- coding: utf-8 -*-

import sys
from purple.finance import Trade
from collections import deque
from numpy import std, mean

PREV_TRADES_USED_FOR_ANALYSIS = 200
TRADE_COUNT_REQUIRED_FOR_ANALYSIS = 20
VOLUME_SPIKE_PERCENTAGE = 1.05

class AnomalousTradeFinder:
    def __init__(self):
        self.pricedelta_history = {}

    def is_anomalous(self, trade):
        self._add(trade)

        # If we've got enough trades, start to analyse
        if len(self.pricedelta_history[trade.symbol]) >= TRADE_COUNT_REQUIRED_FOR_ANALYSIS:
            
            return self._is_volume_spike(trade) or self._is_fat_finger(trade)

    def _is_fat_finger(self,trade):

        current_delta = self.pricedelta_history[trade.symbol][-1]["delta"]
        previous_mean = self.pricedelta_history[trade.symbol][-2]["mean"]
        previous_stdev = self.pricedelta_history[trade.symbol][-2]["stdev"]

        if current_delta >= previous_mean + previous_stdev * 3:
            # We don't use stdev or mean of anomalous trades
            self.pricedelta_history[trade.symbol].pop()["delta"]
            return True
        return False

    def _is_volume_spike(self,trade):
        # If this is our nth trade, analyse
        if self.pricedelta_history[trade.symbol][-1]["count"] == PREV_TRADES_USED_FOR_ANALYSIS:
            self.pricedelta_history[trade.symbol][-1]["count"] = 0
            previous_volume = self.pricedelta_history[trade.symbol][-1]["previous_volume"]
            current_volume = self.pricedelta_history[trade.symbol][-1]["current_volume"]

            # Reset our previous and current volumes
            self.pricedelta_history[trade.symbol][-1]["previous_volume"] = current_volume
            self.pricedelta_history[trade.symbol][-1]["current_volume"] = 0
            if previous_volume != 0:
                # If the volume is above a set percentage, reset values and return true
                return previous_volume * VOLUME_SPIKE_PERCENTAGE <= current_volume

    def _add(self, trade):
        if trade.symbol not in self.pricedelta_history:

            self.pricedelta_history[trade.symbol] = deque([{
                'price': trade.price,
                'delta': 0,
                'current_volume': trade.size,
                'previous_volume': 0,
                'count': 1,
                'mean': 0,
                'stdev': 0,
            }], PREV_TRADES_USED_FOR_ANALYSIS)
        else:
            delta = trade.price - self.pricedelta_history[trade.symbol][-1]["price"]
            count = self.pricedelta_history[trade.symbol][-1]["count"] + 1
            volume = self.pricedelta_history[trade.symbol][-1]["current_volume"] + trade.size
            prev_volume = self.pricedelta_history[trade.symbol][-1]["previous_volume"]

            self.pricedelta_history[trade.symbol].append({
                'price': trade.price,
                'delta': delta,
                'current_volume': volume,
                'previous_volume': prev_volume,
                'count': count,
            })

            deltas = [x["delta"] for x in self.pricedelta_history[trade.symbol]]
            self.pricedelta_history[trade.symbol][-1]["mean"] = mean(deltas)
            self.pricedelta_history[trade.symbol][-1]["stdev"] = std(deltas)
