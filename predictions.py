import math,datetime, time
from purple.finance import Trade

#symbol is a list of trades
def predict(symbol):
    #create lists of x/y values (times/prices)
    times = []
    prices = []
    timeAvg = 0.0
    priceAvg = 0.0
    numTrades = 0
    for trade in symbol:
        time = time.mktime(trade.time.timetuple())
        times.append(time)
        prices.append(trade.price)
        timeAvg += time
        priceAvg += trade.price
        numTrades += 1
    timeAvg /= numTrades
    priceAvg /= numTrades

    #calculate average time gap
    timeGap = 0.0
    for i in range(1,numTrades):
        timeGap += (times[i]-times[i-1])
    timeGap /= numTrades
    
    #gradient, intercept and standard deviation calculation
    numerator = 0.0
    denominator = 0.0
    priceDev = 0.0
    for i in range(numTrades):
        numerator += (times[i]-timeAvg) * (prices[i]-priceAvg)
        denominator += (times[i]-timeAvg) ** 2
        priceDev += (prices[i]-priceAvg)**2
    m = numerator / denominator
    c = priceAvg - (m*timeAvg)
    stDev = math.sqrt(priceDev/numTrades)

    #Find co-ordinates for each prediction line
    DISTANCE = 10
    xVal = times[numTrades-1]
    #datetime conversion
    startTime = datetime.datetime.fromtimestamp(xVal)
    initialTime = datetime.datetime.fromtimestamp(times[0])
    startPoint = (startTime,m*times[numTrades-1]+c)

    predictionLines = [[(initialTime,m*times[0]+c),startPoint],
                       [startPoint], #min3
                       [startPoint], #min2
                       [startPoint], #min1
                       [startPoint], #0
                       [startPoint], #max1
                       [startPoint], #max2
                       [startPoint]] #max3
    for p in range(-3,4):
        for i in DISTANCE:
            xVal += timeGap
            xValTime = datetime.datetime.fromtimestamp(xVal)
            predictionLines[p+4].append((xValTime,m*xVal+c+((math.sqrt(i)*p)*stDev)))
        xVal = times[numTrades-1]
    return (predictionLines)

    #Prediction line consists of start and end co-ordinates of:
    #[trendline, 3rdDevDown, 2ndDevDown, 1stDevDown, trendlineCont, 1stDevUp, 2ndDevUp, 3rdDevUp]
