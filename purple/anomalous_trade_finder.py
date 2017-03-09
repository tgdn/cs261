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
                'volume': trade.size,
                'bid_ask_spread': trade.ask - trade.bid
            }]
        else:
            price_delta = trade.price - self.trade_history[trade.symbol][-1]["price"]
            self.trade_history[trade.symbol].append({
                'time': trade.time,
                'id': identifier,
                'price': trade.price,
                # Round to stop float errors
                'price_delta': round(price_delta,3),
                'volume': trade.size,
                'bid_ask_spread': trade.ask - trade.bid
            })

        if trade.symbol not in self.stats:
            self.stats[trade.symbol] = {
                'trade_count_per_min': 1,
                'minutes': 1,
                'prev_minutes_total_trades': 0,
                'current_minute': trade.time.strftime("%M"),
                'current_hour': trade.time.strftime("%H"),
                'hourly_vol': [0],
                'hourly_max_change':[0],
                'hourly_max': trade.price,
                'hourly_min': trade.price
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
            prices = [x["price"] for x in self.trade_history[key]]

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
                'hourly_vol': self.stats[key]["hourly_vol"],
                'hourly_max_change': self.stats[key]["hourly_max_change"],
                'hourly_max': self.stats[key]["hourly_max"],
                'hourly_min': self.stats[key]["hourly_min"],
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
            #Keep track of index in array of hourly volume sums
            index_pointer = 0
            vol_counter = 0

            for time in times:
                #Calculate statistics for db table
                if self._calculate_trades_per_min(time, trade_count, key):
                    trade_count = 0
                trade_count += 1

                #Calculate volumes for every hour
                if self.stats[key]["current_hour"] != time.strftime("%H"):
                    self.stats[key]["current_hour"] = time.strftime("%H")
                    self.stats[key]["hourly_vol"].append(0)
                    self.stats[key]["hourly_max_change"][index_pointer] = self.stats[key]["hourly_max"] - self.stats[key]["hourly_min"]
                    self.stats[key]["hourly_max_change"].append(0)
                    #Reset current min and max with first trade of new hour
                    self.stats[key]["hourly_max"] = prices[vol_counter + 1]
                    self.stats[key]["hourly_min"] = prices[vol_counter + 1]
                    index_pointer += 1
                else:
                    self.stats[key]["hourly_vol"][index_pointer] += volumes[vol_counter]
                    if prices[vol_counter] > self.stats[key]["hourly_max"]:
                        self.stats[key]["hourly_max"] = prices[vol_counter]
                    if prices[vol_counter] < self.stats[key]["hourly_min"]:
                        self.stats[key]["hourly_min"] = prices[vol_counter]

                #Check for bid ask spread errors
                if self.trade_history[key][vol_counter]["bid_ask_spread"] < 0:
                    description = 'Negative bid ask spread for ' + key
                    self.add_anomaly(ids[vol_counter], times[vol_counter], description, 'NBAS', 1, key)
                    
                vol_counter += 1

            #Check for volume spikes
            self._calculate_vol_spikes(key)
            #Update statsistics for db
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

    def _calculate_vol_spikes(self, key):
        # First work out the mean and standard deviation for every hour of volume sums
        mean_vol = mean(self.stats[key]["hourly_vol"])
        vol_stdev = std(self.stats[key]["hourly_vol"])

        # Work out mean and stdev of maximum hourly price change
        mean_max_price_change = mean(self.stats[key]["hourly_max_change"])
        max_price_change_stdev = std(self.stats[key]["hourly_max_change"])

        spike = False
        # Iterate through each hourly volume sum and check if it's outside of mean + n * stdev
        index = 0

        sev = -1

        for volume in self.stats[key]["hourly_vol"]:
            if volume >= mean_vol + 5 * vol_stdev:
                spike = True
                description = 'Hourly volume spike from ' + str(index + 1) + ' to ' + str(index + 2) + ' for ' + key
                self.add_anomaly(-1, index + 1, description, 'VS', 1, key)
                sev = 1
            elif volume >= mean_vol + 4 * vol_stdev:
                spike = True
                description = 'Hourly volume spike from ' + str(index + 1) + ' to ' + str(index + 2) + ' for ' + key
                self.add_anomaly(-1, index + 1, description, 'VS', 2, key)
                sev = 2
            elif volume >= mean_vol + 3 * vol_stdev:
                spike = True
                description = 'Hourly volume spike from ' + str(index + 1) + ' to ' + str(index + 2) + ' for ' + key
                self.add_anomaly(-1, index + 1, description, 'VS', 3, key)
                sev = 3
            if spike:
                self._calculate_pump_bear(key, index, mean_max_price_change, max_price_change_stdev, sev)
            index += 1

    def _calculate_pump_bear(self, key, hour, mean, stdev, sev):
        if self.stats[key]["hourly_max_change"][hour] > mean +  2 * stdev:
            description = 'Hourly pump and dump/bear raid from ' + str(hour + 1) + ' to ' + str(hour + 2) + ' for ' + key
            self.add_anomaly(-1, hour + 1, description, 'PDBR', sev, key)
            

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

        if new_delta_to_add >= delta_values["stdev"] * 7 + delta_values["mean"]:
            description = 'Fat finger error on price for ' + trade.symbol
            self.add_anomaly(identifier, trade.time, description, 'FFP', 1, trade.symbol)
        elif new_delta_to_add >= delta_values["stdev"] * 6 + delta_values["mean"]:
            #Alert fat finger on price
            description = 'Fat finger error on price for ' + trade.symbol
            self.add_anomaly(identifier, trade.time, description, 'FFP', 2, trade.symbol)
        elif new_delta_to_add >= delta_values["stdev"] * 5 + delta_values["mean"]:
            #Alert fat finger on price
            description = 'Fat finger error on price for ' + trade.symbol
            self.add_anomaly(identifier, trade.time, description, 'FFP', 3, trade.symbol)


        vol_values = self.welford(trade_count, vol_stdev, vol_mean, new_vol_to_add)


        if new_vol_to_add >= vol_values["stdev"] * 7 + vol_values["mean"]:
            #Alert fat finger on volume
            description = 'Fat finger error on volume for ' + trade.symbol
            self.add_anomaly(identifier, trade.time, description, 'FFV', 1, trade.symbol)
        elif new_vol_to_add >= vol_values["stdev"] * 6 + vol_values["mean"]:
            #Alert fat finger on volume
            description = 'Fat finger error on volume for ' + trade.symbol
            self.add_anomaly(identifier, trade.time, description, 'FFV', 2, trade.symbol)
        elif new_vol_to_add >= vol_values["stdev"] * 5 + vol_values["mean"]:
            #Alert fat finger on volume
            description = 'Fat finger error on volume for ' + trade.symbol
            self.add_anomaly(identifier, trade.time, description, 'FFV', 3, trade.symbol)

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
            price_query = "(SELECT MAX(price) FROM trades WHERE symbol_name=\'" + key + "\' AND analysis_date=\'" + date + "\' LIMIT 1) "
            price_query += "UNION (SELECT MIN(price) FROM trades WHERE symbol_name=\'" + key + "\' AND analysis_date=\'" + date + "\' LIMIT 1)"

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
            max_price = prices.fetchone()[0]
            min_price = prices.fetchone()[0]
            price_change_to_add = min_price - max_price
            spike = False
            #Calculate new mean and standard deviation for day's price change
            new_day_change_mean_stdev = self.welford(day_count, day_price_change_stdev, day_price_change_mean, price_change_to_add)

            if to_add >= new_vol_stdev_mean["mean"] + 7 * new_vol_stdev_mean["stdev"]:
                spike = True
                #Alert volume spike for day
                description = 'Volume spike over past day for ' + key
                self.add_anomaly(date, -1, description, 'VS', 1, key)
            elif to_add >= new_vol_stdev_mean["mean"] + 6 * new_vol_stdev_mean["stdev"]:
                spike = True
                #Alert volume spike for day
                description = 'Volume spike over past day for ' + key
                self.add_anomaly(date, -1, description, 'VS', 2, key)
            elif to_add >= new_vol_stdev_mean["mean"] + 5 * new_vol_stdev_mean["stdev"]:
                spike = True
                #Alert volume spike for day
                description = 'Volume spike over past day for ' + key
                self.add_anomaly(date, -1, description, 'VS', 3, key)
            # If there's a volume spike, check for a pump and dump
            if spike:
                #Pump and dump if price change is outside of n * stdev + mean
                if price_change_to_add >= new_day_change_mean_stdev["mean"] + 7 * new_day_change_mean_stdev["stdev"]:
                    description = 'Pump and dump/bear raid over past day for ' + key
                    self.add_anomaly(date, -1, description, 'PDBR', 1, key)
                elif price_change_to_add >= new_day_change_mean_stdev["mean"] + 6 * new_day_change_mean_stdev["stdev"]:
                    description = 'Pump and dump/bear raid over past day for ' + key
                    self.add_anomaly(date, -1, description, 'PDBR', 2, key)
                elif price_change_to_add >= new_day_change_mean_stdev["mean"] + 5 * new_day_change_mean_stdev["stdev"]:
                    description = 'Pump and dump/bear raid over past day for ' + key
                    self.add_anomaly(date, -1, description, 'PDBR', 3, key)

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
                description = 'Fat finger error on price for ' + key
                self.add_anomaly(ids[counter], times[counter], description, 'FFP', 1, key)
            elif value >= self.stats[key]["delta_mean"] + 6 * self.stats[key]["delta_stdev"] or value <= (self.stats[key]["delta_mean"] - 6 * self.stats[key]["delta_stdev"]):
                description = 'Fat finger error on price for ' + key
                self.add_anomaly(ids[counter], times[counter], description, 'FFP', 2, key)
            elif value >= self.stats[key]["delta_mean"] + 5 * self.stats[key]["delta_stdev"] or value <= (self.stats[key]["delta_mean"] - 5 * self.stats[key]["delta_stdev"]):
                description = 'Fat finger error on price for ' + key
                self.add_anomaly(ids[counter], times[counter], description, 'FFP', 3, key)
            counter += 1
        counter = 0
        for volume in volumes:
            if volume >= self.stats[key]["vol_mean"] + 7 * self.stats[key]["vol_stdev"]:
                description = 'Fat finger error on volume ' + key
                self.add_anomaly(ids[counter], times[counter], description, 'FFV', 1, key)
            elif volume >= self.stats[key]["vol_mean"] + 6 * self.stats[key]["vol_stdev"]:
                description = 'Fat finger error on volume ' + key
                self.add_anomaly(ids[counter], times[counter], description, 'FFV', 2, key)
            elif volume >= self.stats[key]["vol_mean"] + 5 * self.stats[key]["vol_stdev"]:
                description = 'Fat finger error on volume ' + key
                self.add_anomaly(ids[counter], times[counter], description, 'FFV', 3, key)
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

    def add_anomaly(self, identifier, time, description, error_code, severity, symbol):
        self.anomalous_trades.append({
            'id': identifier,
            'time': time,
            'description': description,
            'error_code': error_code,
            'severity': severity,
            'symbol': symbol
        })
        return True
        


