# -*- coding: utf-8 -*-
# Store time in trades thing

import sys
from purple.finance import Trade
from collections import deque
from numpy import std, mean
from datetime import datetime, timedelta
from purple import db
import math
import pytz

tz = pytz.timezone('Europe/London')

class AnomalousTradeFinder:
    def __init__(self):
        self.trade_history = {}
        self.anomalous_trades = []
        #stats stores the current delta mean+stdev, vol mean + stdev, along with delta for a whole day mean + stdev, and vol for a whole day mean + stdev
        self.stats = {}
        #store the previous trade for each symbol so we can get price deltas
        self.prev_trades = {}
        self.update_characteristics_count = 0

    # Stores relevant information about trades in a dictionary of a list of dictionaries
    def add(self, trade, identifier):
        if trade.symbol not in self.trade_history:
            self.trade_history[trade.symbol] = [{
                'time': trade.time,
                'id': identifier,
                'price': trade.price,
                'price_delta': 0,
                'volume': trade.size
            }]
        else:
            price_delta = trade.price - self.trade_history[trade.symbol][-1]["price"]
            self.trade_history[trade.symbol].append({
                'time': trade.time,
                'id': identifier,
                'price': trade.price,
                'price_delta': price_delta,
                'volume': trade.size
            })

    #This calculates the values after a CSV or the first day of stream data
    def calculate_anomalies_first_day(self, csv):
        self.anomalous_trades = []
        for key in self.trade_history:
            volumes = [x["volume"] for x in self.trade_history[key]]
            deltas = [x["price_delta"] for x in self.trade_history[key]]
            ids = [x["id"] for x in self.trade_history[key]]
            times = [x["time"] for x in self.trade_history[key]]

            #So here we want to add something that stores the total delta for the day, the total volume for day
            #and their mean and stdevs. work out how to handle vol spikes and pump and dump
            self.stats[key] = {
                'delta_mean': mean(deltas),
                'delta_stdev': std(deltas),
                'vol_mean': mean(volumes),
                'vol_stdev': std(volumes),
                'trade_count': len(volumes),
                'total_vol_stdev': 0,
                'total_vol_mean': sum(volumes),
                'day_price_change_mean': self.trade_history[key][-1]["price"] - self.trade_history[key][0]["price"],
                'day_price_change_stdev': 0,
                'day_count': 1
            }

            self.calculate_fat_finger(volumes, deltas, ids, times, key)

            #Decide how to handle vol spikes and pump and dump for one day/csv

            #Get the price of the last added trade for that symbol
            self.prev_trades[key] = self.trade_history[key][-1]["price"]

            #We only want to update the characteristics if we're looking at feed data
            if not csv:
                self.update_characteristics(key)

        if not csv:
            db.session.commit()

        self.trade_history = {}

        return self.anomalous_trades

    #We call this when analysing a trade from the stream that isn't from the first day
    def calculate_anomalies_single_trade(self, trade, identifier):
        self.anomalous_trades = []

        price_delta_mean = self.stats[trade.symbol]["delta_mean"]
        price_delta_stdev = self.stats[trade.symbol]["delta_stdev"]

        vol_mean = self.stats[trade.symbol]["vol_mean"]
        vol_stdev = self.stats[trade.symbol]["vol_stdev"]
        trade_count = self.stats[trade.symbol]["trade_count"] + 1

        new_delta_to_add = trade.price - self.prev_trades[trade.symbol]
        new_vol_to_add = trade.size

        #Calculate new stdev for price deltas
        delta_values = self.welford(trade_count, price_delta_stdev, price_delta_mean, new_delta_to_add)

        if new_delta_to_add >= delta_values["stdev"] * 3 + delta_values["mean"]:
            #Alert fat finger on price
            print "Found a fat finger on price on individual trade"
            self.anomalous_trades.append({
                    'id': identifier,
                    'time': trade.time,
                    'description': 'Fat finger error on price for ' + trade.symbol
            })
        vol_values = self.welford(trade_count, vol_stdev, vol_mean, new_vol_to_add)
        if new_vol_to_add >= vol_values["stdev"] * 3 + vol_values["mean"]:
            #Alert fat finger on volume
            print "Found a fat finger on volume on individual trade"
            self.anomalous_trades.append({
                    'id': identifier,
                    'time': trade.time,
                    'description': 'Fat finger error on volume for ' + trade.symbol
            })

        #Update stats with new statistical values
        self.stats[trade.symbol]['delta_mean'] = delta_values['mean']
        self.stats[trade.symbol]['delta_stdev'] = delta_values['stdev']
        self.stats[trade.symbol]['vol_mean'] = vol_values['mean']
        self.stats[trade.symbol]['vol_stdev'] = vol_values['stdev']
        self.stats[trade.symbol]['trade_count'] = trade_count + 1

        self.update_characteristics(trade.symbol)
        self.update_characteristics_count += 1

        if self.update_characteristics_count == 50:
            db.session.commit()
            self.update_characteristics_count = 0

        return self.anomalous_trades
        
    #Recalculates an existing standard deviation with another added point using Welford's method
    def welford(self, count, stdev, mean, to_add):
        m2 = (stdev ** 2) * (count - 1)
        change = to_add - mean
        mean += change / float(count)
        second_change = to_add - mean
        m2 += change * second_change

        return {
            "stdev": math.sqrt(m2 / float(count)),
            "mean": mean
        }

    #This is when we've just finished a day of trades and we want to find out vol spikes/dips and pump dump or bear raid
    def calculate_anomalies_end_of_day(self, date):
        self.anomalous_trades = []
        for key in self.prev_trades:
            date = (datetime.strptime(date,'%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            vol_query = "SELECT SUM(size) FROM trades WHERE symbol_name=\'" + key + "\' AND analysis_date=\'" + date + "\'"
            volume = db.engine.execute(vol_query)

            #query that gets price change for day
            price_query = "(SELECT price FROM trades WHERE symbol_name=\'" + key + "\' AND analysis_date=\'" + date + "\' ORDER BY id ASC LIMIT 1) "
            price_query += "UNION (SELECT price FROM trades WHERE symbol_name=\'" + key + "\' AND analysis_date=\'" + date + "\' ORDER BY id DESC LIMIT 1)"

            prices = db.engine.execute(price_query)

            day_count = self.stats[key]["day_count"]
            total_vol_stdev = self.stats[key]["total_vol_stdev"]
            total_vol_mean = self.stats[key]["total_vol_mean"]

            day_price_change_mean = self.stats[key]["day_price_change_mean"]
            day_price_change_stdev = self.stats[key]["day_price_change_stdev"]

            to_add = volume.fetchone()[0]

            #Calculate new mean and standard deviation for day's volume
            new_vol_stdev_mean = self.welford(day_count, total_vol_stdev, total_vol_mean, float(to_add))
           
            #Get price change for whole day
            opening_price = prices.fetchone()[0]
            closing_price = prices.fetchone()[0]
            price_change_to_add = closing_price - opening_price

            #Calculate new mean and standard deviation for day's price change
            new_day_change_mean_stdev = self.welford(day_count, day_price_change_stdev, day_price_change_mean, price_change_to_add)

            if to_add >= new_vol_stdev_mean["mean"] + 3 * new_vol_stdev_mean["stdev"]:
                #Alert volume spike for day
                self.anomalous_trades.append({
                    'id': date,
                    'time': -1,
                    'description': 'Volume spike over past day for ' + key 
                })
                
                #Pump and dump if price change is outside of 3stdev + mean
                if price_change_to_add >= new_day_change_mean_stdev["mean"] + 3 * new_day_change_mean_stdev["stdev"]:
                    self.anomalous_trades.append({
                        'id': date,
                        'time': -1,
                        'description': 'Pump and dump over past day for ' + key
                    })

                #Bear raid if price change is outside of 3stdev + mean
                if price_change_to_add <= new_day_change_mean_stdev["mean"] - 3 * new_day_change_mean_stdev["stdev"]:
                    self.anomalous_trades.append({
                        'id': date,
                        'time': -1,
                        'description': 'Bear raid over past day for ' + key
                    })

            #Update stats with new total vol stdev and mean, and new count of days
            self.stats[key]['total_vol_stdev'] = new_vol_stdev_mean['stdev']
            self.stats[key]['total_vol_mean'] = new_vol_stdev_mean['mean']
            self.stats[key]['day_price_change_mean'] = new_day_change_mean_stdev['mean']
            self.stats[key]['day_price_change_stdev'] = new_day_change_mean_stdev['stdev']
            self.stats[key]['day_count'] = self.stats[key]['day_count'] + 1

            self.update_characteristics(key)

        db.session.commit()
        return self.anomalous_trades

    #Calculate fat finger errors on volume and price, add every one to anomalous_trades
    def calculate_fat_finger(self, volumes, deltas, ids, times, key):
        counter = 0
        for value in deltas:
            if value >= self.stats[key]["delta_mean"] + 3 * self.stats[key]["delta_stdev"] or value <= (self.stats[key]["delta_mean"] - 3 * self.stats[key]["delta_stdev"]):
                self.anomalous_trades.append({
                    'id': ids[counter],
                    'time': times[counter],
                    'description': 'Fat finger error on price for ' + key 
                })
            counter += 1
        counter = 0
        for volume in volumes:
            if volume >= self.stats[key]["vol_mean"] + 3 * self.stats[key]["vol_stdev"] or volume <= (self.stats[key]["vol_mean"] - 3 * self.stats[key]["vol_stdev"]):
                self.anomalous_trades.append({
                    'id': ids[counter],
                    'time': times[counter],
                    'description': 'Fat finger error on volume ' + key
                })
            counter += 1

    def update_characteristics(self, symbol):
        to_insert = {
            'average_volume': self.stats[symbol]["vol_mean"],
            'average_daily_volume': self.stats[symbol]["total_vol_mean"],
            'average_price_change_daily': self.stats[symbol]["day_price_change_mean"],
            'average_price_change': self.stats[symbol]["delta_mean"],
            'timestamp': tz.localize(datetime.now())
        }

        db.session.query(db.SymbolModel).filter_by(name=symbol).update(to_insert)
        


