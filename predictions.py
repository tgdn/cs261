import math
#from purple.finance import Trade

#symbol is a list of trades
def predict(symbol):
    #create lists of x/y values (times/prices)
    times = []
    prices = []
    timeAvg = 0.0
    priceAvg = 0.0
    numTrades = 0
    for trade in symbol:
        times.append(trade[0])
        prices.append(trade[1])
        timeAvg += trade[0]
        priceAvg += trade[1]
        numTrades += 1
    timeAvg /= numTrades
    priceAvg /= numTrades

    timeGap = 0.0
    priceGap = 0.0
    for i in range(1,numTrades):
        timeGap += (times[i]-times[i-1])
        priceGap += (prices[i]-prices[i-1])**2
    timeGap /= numTrades
    stDev = math.sqrt(priceGap/numTrades)
    
    #gradient and intercept calculation
    numerator = 0.0
    denominator = 0.0
    for i in range(numTrades):
        numerator += (times[i]-timeAvg) * (prices[i]-priceAvg)
        denominator += (times[i]-timeAvg) ** 2
    m = numerator / denominator
    c = priceAvg - (m*timeAvg)
    
##    #r squared calculation            UNCOMMENT FOR R2 REFINEMENT
##    trendlineErr = 0.0
##    priceErr = 0.0
##    for i in range(numTrades):
##        priceErr += (prices[i]-priceAvg) ** 2
##        trendlineErr += ((m*times[i]+c)-priceAvg) ** 2
##    r2 = trendlineErr/priceErr

    #Find co-ordinates for each prediction line
    DISTANCE = 10
    xVal = times[numTrades-1] + timeGap
    yVal = m*xVal+c
    xValLast = xVal + timeGap*DISTANCE
    yValLast = m*xValLast+c

   ##DISTANCE *= (1-r2)         UNCOMMENT FOR R2 REFINEMENT
    
    predictionLines = [[(times[0],m*times[0]+c),(times[numTrades-1],m*times[numTrades-1]+c)],
        [(xVal,yVal + 1*stDev),(xValLast,yValLast + DISTANCE*(1*stDev))],
        [(xVal,yVal + 2*stDev),(xValLast,yValLast + DISTANCE*(2*stDev))],
        [(xVal,yVal + 3*stDev),(xValLast,yValLast + DISTANCE*(3*stDev))],
        [(xVal,yVal - 1*stDev),(xValLast,yValLast - DISTANCE*(1*stDev))],
        [(xVal,yVal - 2*stDev),(xValLast,yValLast - DISTANCE*(2*stDev))],
        [(xVal,yVal - 3*stDev),(xValLast,yValLast - DISTANCE*(3*stDev))]
    ]
    
    return (predictionLines)

    #Prediction line consists of start and end co-ordinates of:
    #[trendline, 1stDevUp, 2ndDevUp, 3rdDevUp, 1stDevDown, 2ndDevDown, 3rdDevDown]
