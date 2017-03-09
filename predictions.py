import math,datetime, time
from purple.finance import Trade

# Symbol is a list of trades requested by the frontend
def predict(symbol):
    # Create lists of x/y values (times/prices) along with averages
    times = []
    prices = []
    timeAvg = 0.0
    priceAvg = 0.0
    numTrades = 0
    # Iterate throught the list symbol
    for trade in symbol:
        # Convert time to a number used for mathematical methods
        time = time.mktime(trade.time.timetuple())
        # Add times and prices to the lists...
        times.append(time)
        prices.append(trade.price)
        # ...and then add them to the averages
        timeAvg += time
        priceAvg += trade.price
        numTrades += 1
    # Calculate averages
    timeAvg /= numTrades
    priceAvg /= numTrades

    # Calculate average time gap
    timeGap = 0.0
    for i in range(1,numTrades):
        timeGap += (times[i]-times[i-1])
    timeGap /= numTrades
    
    # Gradient, intercept and standard deviation calculation
    numerator = 0.0
    denominator = 0.0
    priceDev = 0.0
    # Totals calculated by iterating over each trades
    for i in range(numTrades):
        numerator += (times[i]-timeAvg) * (prices[i]-priceAvg)
        denominator += (times[i]-timeAvg) ** 2
        priceDev += (prices[i]-priceAvg)**2
    # m is the gradient of the trendline
    m = numerator / denominator
    # c is the intercept of the trendline
    c = priceAvg - (m*timeAvg)
    stDev = math.sqrt(priceDev/numTrades)

    # Find co-ordinates for each prediction line
    # DISTANCE is the number of trades away to predict
    DISTANCE = 10
    xVal = times[numTrades-1]
    # The numbers used for time are converted back to the datetime format
    startTime = datetime.datetime.fromtimestamp(xVal)
    initialTime = datetime.datetime.fromtimestamp(times[0])
    startPoint = (startTime,m*times[numTrades-1]+c)

    # The final value to return is initialized, holding the trendline and the startpoint of each prediction line
    predictionLines = [[(initialTime,m*times[0]+c),startPoint],
                       [startPoint], #min3
                       [startPoint], #min2
                       [startPoint], #min1
                       [startPoint], #0
                       [startPoint], #max1
                       [startPoint], #max2
                       [startPoint]] #max3
    # Append the prediction line values to each line
    for p in range(-3,4):
        for i in DISTANCE:
            # Calculate the point at each x value up to DISTANCE trades away and append it to the respective line array
            xVal += timeGap
            xValTime = datetime.datetime.fromtimestamp(xVal)
            predictionLines[p+4].append((xValTime,m*xVal+c+((math.sqrt(i)*p)*stDev)))
        xVal = times[numTrades-1]
    return (predictionLines)

    # Prediction line consists of start and end co-ordinates of:
    # [trendline, 3rdDevDown, 2ndDevDown, 1stDevDown, trendlineContinued, 1stDevUp, 2ndDevUp, 3rdDevUp]
