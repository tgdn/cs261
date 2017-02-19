# -*- coding: utf-8 -*-

from purple.finance import Trade
from collections import deque
from numpy import std, mean

PREV_TRADES_USED_FOR_ANALYSIS = 200
TRADE_COUNT_REQUIRED_FOR_ANALYSIS = 20

class AnomalousTradeFinder:
    def __init__(self):
        self.pricedelta_history = {}

    def is_anomalous(self, trade):
        self._add(trade)

        if len(self.pricedelta_history[trade.symbol]) >= TRADE_COUNT_REQUIRED_FOR_ANALYSIS:
            current_delta = self.pricedelta_history[trade.symbol][-1]["delta"]
            previous_mean = self.pricedelta_history[trade.symbol][-2]["mean"]
            previous_stdev = self.pricedelta_history[trade.symbol][-2]["stdev"]

            if current_delta >= previous_mean + previous_stdev * 3:
                #We don't use stdev or mean of anomalous trades
                self.pricedelta_history[trade.symbol].pop()["delta"]
                return True

        return False

    def _add(self, trade):
        if trade.symbol not in self.pricedelta_history:

            self.pricedelta_history[trade.symbol] = deque([{
                'price': trade.price,
                'delta': 0,
                'mean': 0,
                'stdev': 0
            }], PREV_TRADES_USED_FOR_ANALYSIS)
        else:
            delta = trade.price - self.pricedelta_history[trade.symbol][-1]["price"]

            self.pricedelta_history[trade.symbol].append({
                'price': trade.price,
                'delta': delta,
            })

            deltas = [x["delta"] for x in self.pricedelta_history[trade.symbol]]
            self.pricedelta_history[trade.symbol][-1]["mean"] = mean(deltas)
            self.pricedelta_history[trade.symbol][-1]["stdev"] = std(deltas)
