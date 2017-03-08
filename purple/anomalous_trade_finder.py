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
VOLUME_SPIKE_PERCENTAGE = 1.1

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
                # Round to stop float errors
                'price_delta': round(price_delta,3),
                'volume': trade.size
            })

        if trade.symbol not in self.stats:
            self.stats[trade.symbol] = {
                'trade_count_per_min': 1,
                'minutes': 1,
                'prev_minutes_total_trades': 0,
                'current_minute': trade.time.strftime("%M"),
                'current_hour': trade.time.strftime("%H"),
                'current_hour_vol_sum': trade.size,
                'prev_hour_vol_sum': 0
            }
        else:
            self.stats[trade.symbol]["trade_count_per_min"] += 1

    #This calculates the values after a CSV or the first day of stream data
    def calculate_anomalies_first_day(self, csv):
        self.anomalous_trades = []
        for key in self.trade_history:
            volumes = [x["volume"] for x in self.trade_history[key]]
            deltas = [x["price_delta"] for x in self.trade_history[key]]
            ids = [x["id"] for x in self.trade_history[key]]
            times = [x["time"] for x in self.trade_history[key]]

            #Get the price of the last added trade for that symbol
            self.prev_trades[key] = self.trade_history[key][-1]["price"]

            #So here we want to add something that stores the total delta for the day, the total volume for day
            #and their mean and stdevs.
            self.stats[key] = {
                'trade_count_per_min': self.stats[key]["trade_count_per_min"],
                'minutes': self.stats[key]["minutes"],
                'prev_minutes_total_trades': self.stats[key]["prev_minutes_total_trades"],
                'current_minute': self.stats[key]["current_minute"],
                'current_hour': self.stats[key]["current_hour"],
                'current_hour_vol_sum': self.stats[key]["current_hour_vol_sum"],
                'prev_hour_vol_sum': self.stats[key]["prev_hour_vol_sum"],
                'delta_mean': mean(deltas),
                'delta_stdev': std(deltas),
                'vol_mean': mean(volumes),
                'vol_stdev': std(volumes),
                'trade_count': len(volumes),
                'total_vol_stdev': 0,
                'total_vol_mean': sum(volumes),
                'day_price_change_mean': self.trade_history[key][-1]["price"] - self.trade_history[key][0]["price"],
                'day_price_change_stdev': 0,
                'day_count': 1,
                'price_change_percentage': (self.trade_history[key][-1]["price"] / float(self.trade_history[key][-2]["price"]))
            }

            self.calculate_fat_finger(volumes, deltas, ids, times, key)

            #Decide how to handle pump and dump for one day/csv

            #We only want to update the characteristics if we're looking at feed data
            current_minute = times[0]
            trade_count = 1
            index_pointer = 0

            for time in times:
                #Calculate statistics
                if self._calculate_trades_per_min(time, trade_count, key):
                    trade_count = 0
                trade_count += 1
                """
                #Check for volume spikes in past hour
                if self._calculate_vol_spike_per_hour(time, volumes, index_pointer, key):
                    self.anomalous_trades.append({
                        'id': index_pointer,
                        'time': times[index_pointer],
                        'description': 'Volume spike for ' + key,
                        'error_code': 'VS'
                    })
                index_pointer += 1
                """
            self.update_characteristics(key)

            db.session.commit()

        self.trade_history = {}

        return self.anomalous_trades

    def _calculate_trades_per_min(self, time, trade_count, key):
        if time.strftime("%M") != self.stats[key]["current_minute"]:
            self.stats[key]["trade_count_per_min"] = (self.stats[key]["prev_minutes_total_trades"] + trade_count) / float(self.stats[key]["minutes"])
            self.stats[key]["minutes"] += 1
            self.stats[key]["prev_minutes_total_trades"] += trade_count
            self.stats[key]["current_minute"] = time.strftime("%M")
            return True
        return False

    def _calculate_vol_spike_per_hour(self, time, volumes, count, key):
        self.stats[key]["current_hour_vol_sum"] += volumes[count]
        current_hour_vol_sum = self.stats[key]["current_hour_vol_sum"]
        current_hour = self.stats[key]["current_hour"]
        prev_hour_vol_sum = self.stats[key]["prev_hour_vol_sum"]

        #Update values
        self.stats[key]["prev_hour_vol_sum"] = self.stats[key]["current_hour_vol_sum"]
        self.stats[key]["current_hour_vol_sum"] = 0
        self.stats[key]["current_hour"] = time.strftime("%H")

        if time.strftime("%H") != current_hour:
            if prev_hour_vol_sum:
                return prev_hour_vol_sum * VOLUME_SPIKE_PERCENTAGE <= current_hour_vol_sum
        return False

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

        if new_delta_to_add >= delta_values["stdev"] * 6 + delta_values["mean"]:
            #Alert fat finger on price
            print "Found a fat finger on price on individual trade"
            self.anomalous_trades.append({
                    'id': identifier,
                    'time': trade.time,
                    'description': 'Fat finger error on price for ' + trade.symbol,
                    'error_code': 'FFP',
                    'severity': 2
            })
        vol_values = self.welford(trade_count, vol_stdev, vol_mean, new_vol_to_add)
        if new_vol_to_add >= vol_values["stdev"] * 6 + vol_values["mean"]:
            #Alert fat finger on volume
            print "Found a fat finger on volume on individual trade"
            self.anomalous_trades.append({
                    'id': identifier,
                    'time': trade.time,
                    'description': 'Fat finger error on volume for ' + trade.symbol,
                    'error_code': 'FFV',
                    'severity': 2
            })

        #Update stats with new statistical values
        self.stats[trade.symbol]["trade_count_per_min"] += 1
        self.stats[trade.symbol]['delta_mean'] = delta_values['mean']
        self.stats[trade.symbol]['delta_stdev'] = delta_values['stdev']
        self.stats[trade.symbol]['vol_mean'] = vol_values['mean']
        self.stats[trade.symbol]['vol_stdev'] = vol_values['stdev']
        self.stats[trade.symbol]['trade_count'] = trade_count + 1
        self.stats[trade.symbol]['price_change_percentage'] = (trade.price / float(self.prev_trades[trade.symbol]))

        self._calculate_trades_per_min(trade.time, self.stats[trade.symbol]["trade_count_per_min"], trade.symbol)
        self.update_characteristics(trade.symbol)
        self.update_characteristics_count += 1

        if self.update_characteristics_count == 50:
            db.session.commit()
            self.update_characteristics_count = 0

        self.prev_trades[trade.symbol] = trade.price

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

            if to_add >= new_vol_stdev_mean["mean"] + 6 * new_vol_stdev_mean["stdev"]:
                #Alert volume spike for day
                self.anomalous_trades.append({
                    'id': date,
                    'time': -1,
                    'description': 'Volume spike over past day for ' + key,
                    'error_code': 'VS',
                    'severity': 2
                })
                
                #Pump and dump if price change is outside of 3stdev + mean
                if price_change_to_add >= new_day_change_mean_stdev["mean"] + 6 * new_day_change_mean_stdev["stdev"]:
                    self.anomalous_trades.append({
                        'id': date,
                        'time': -1,
                        'description': 'Pump and dump over past day for ' + key,
                        'error_code': 'PD',
                        'severity': 2
                    })

                #Bear raid if price change is outside of 3stdev + mean
                if price_change_to_add <= new_day_change_mean_stdev["mean"] - 6 * new_day_change_mean_stdev["stdev"]:
                    self.anomalous_trades.append({
                        'id': date,
                        'time': -1,
                        'description': 'Bear raid over past day for ' + key,
                        'error_code': 'BR',
                        'severity': 2
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
            if value >= self.stats[key]["delta_mean"] + 7 * self.stats[key]["delta_stdev"] or value <= (self.stats[key]["delta_mean"] - 7 * self.stats[key]["delta_stdev"]):
                self.anomalous_trades.append({
                    'id': ids[counter],
                    'time': times[counter],
                    'description': 'Fat finger error on price for ' + key,
                    'error_code': 'FFP',
                    'severity': 1
                }) 
            elif value >= self.stats[key]["delta_mean"] + 6 * self.stats[key]["delta_stdev"] or value <= (self.stats[key]["delta_mean"] - 6 * self.stats[key]["delta_stdev"]):
                self.anomalous_trades.append({
                    'id': ids[counter],
                    'time': times[counter],
                    'description': 'Fat finger error on price for ' + key,
                    'error_code': 'FFP',
                    'severity': 2
                })
            elif value >= self.stats[key]["delta_mean"] + 5 * self.stats[key]["delta_stdev"] or value <= (self.stats[key]["delta_mean"] - 5 * self.stats[key]["delta_stdev"]):
                self.anomalous_trades.append({
                    'id': ids[counter],
                    'time': times[counter],
                    'description': 'Fat finger error on price for ' + key,
                    'error_code': 'FFP',
                    'severity': 3
                })
            counter += 1
        counter = 0
        for volume in volumes:
            if volume >= self.stats[key]["vol_mean"] + 7 * self.stats[key]["vol_stdev"]:
                self.anomalous_trades.append({
                    'id': ids[counter],
                    'time': times[counter],
                    'description': 'Fat finger error on volume ' + key,
                    'error_code': 'FFV',
                    'severity': 1
                })
            elif volume >= self.stats[key]["vol_mean"] + 6 * self.stats[key]["vol_stdev"]:
                self.anomalous_trades.append({
                    'id': ids[counter],
                    'time': times[counter],
                    'description': 'Fat finger error on volume ' + key,
                    'error_code': 'FFV',
                    'severity': 2
                })
            elif volume >= self.stats[key]["vol_mean"] + 5 * self.stats[key]["vol_stdev"]:
                self.anomalous_trades.append({
                    'id': ids[counter],
                    'time': times[counter],
                    'description': 'Fat finger error on volume ' + key,
                    'error_code': 'FFV',
                    'severity': 3
                })
            counter += 1

    def update_characteristics(self, symbol):
        to_insert = {
            'average_volume': self.stats[symbol]["vol_mean"],
            'average_daily_volume': self.stats[symbol]["total_vol_mean"],
            'average_price_change_daily': self.stats[symbol]["day_price_change_mean"],
            'average_price_change': self.stats[symbol]["delta_mean"],
            'average_trades_per_minute': self.stats[symbol]["trade_count_per_min"],
            'last_price_change_percentage': self.stats[symbol]["price_change_percentage"],
            'timestamp': tz.localize(datetime.now())
        }
        db.session.query(db.SymbolModel).filter_by(name=symbol).update(to_insert)
        


