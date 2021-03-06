import warnings

import matplotlib.pyplot as plt
import pandas as p
import numpy as np


from DataRead.StockReader import StockReader
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
import datetime
from dateutil import parser

class PreProcessor:

    class StopLossLim:
        def __init__(self, price):
            self.price = price

    class DataBlock:
        def __init__(self, series, date):
            self.series = series
            self.date = date

        def getRollingDataForDay(self, window, operation='mean'):
            s = self.series.resample("1D").fillna('ffill').rolling(window=window)
            if (operation == 'mean'):
                return s.mean()[self.date]
            if (operation == 'max'):
                return s.max()[self.date]
            if (operation == 'min'):
                return s.min()[self.date]
            if (operation == 'var'):
                return s.var()[self.date]

        def getRollingData(self, window, operation='mean'):
            s = self.series.resample("1D").fillna('ffill').rolling(window=window)
            if (operation == 'mean'):
                return s.mean()
            if (operation == 'max'):
                return s.max()
            if (operation == 'min'):
                return s.min()
            if (operation == 'var'):
                return s.var()

        def peaked(self, distBtwnMinAndMax, dataPieces, minWindow=30, maxWindow=20, peakWidthMax=20, MinToMaxRatio=.9):
            max = self.series.resample("1D").fillna('ffill').rolling(window=maxWindow).max()[self.date]
            min = self.series.resample("1D").fillna('ffill').rolling(window=minWindow).min()[self.date]

            index = 0  # index of current datablock
            for i in range(0, len(dataPieces)):
                if (dataPieces[i].date == self.date):
                    index = i
                    break

            # Peak is tall enough
            diff = min / max
            if diff > MinToMaxRatio:
                return False

            # If is min
            if min == self.open:
                a = False
                dist = index - distBtwnMinAndMax
                for i in range(dist, index):
                    if dataPieces[i].open == max:  # If max was within distBtwnMinAndMax
                        a = True
                if not a:
                    return False
            else:
                return False

            # Peak is skinny enough
            dist = index - distBtwnMinAndMax

            for i in range(dist, index):
                if dataPieces[i].open <= min <= dataPieces[i + 1].open:
                    return True
            return False

    def __init__(self, company, start=datetime.datetime(2014, 1, 1), end=datetime.date.today(), source='Google', local = False):
        # if Local is set to true, will look in the local_csv folder for csv file of name. Will ignore start and end time, and
        # Process all data
        self.start = start
        self.end = end
        self.company = company
        if local == False:
            if source == 'Google':
                url = StockReader.getStockUrlGoogle(company, start, end)
            elif source == 'Yahoo':
                url = StockReader.getStockUrlYahoo(company, start, end)
            else:
                raise ValueError("Invalid source")
            data = p.read_csv(url, parse_dates=True, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        else:
            csvFile = "local_csvs\\" + company + ".csv"
            data = p.read_csv(csvFile, parse_dates=True, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        d = [x for x in data['Date'].values]
        o = [x for x in data['Open'].values]
        h = [x for x in data['High'].values]
        l = [x for x in data['Low'].values]
        c = [x for x in data['Close'].values]

        dates = []
        open = []
        high = []
        low = []
        close = []

        for x in range(1, len(d)):
            try:
                date = parser.parse(d[x])
                opn = float(o[x])
                hgh = float(h[x])
                lw = float(l[x])
                clse = float(c[x])

            except:
                # Skip any issues we have
                continue

            dates.append(date)
            open.append(opn)
            high.append(hgh)
            low.append(lw)
            close.append(clse)

        dataPieces = []
        s = p.Series(open, dates)
        s.cumsum()
        # Initialize all basic data
        for dayIndex in range(0, len(dates)):
            db = PreProcessor.DataBlock(s, dates[dayIndex])
            db.index = dayIndex
            db.open = open[dayIndex]
            db.high = high[dayIndex]
            db.low = low[dayIndex]
            db.close = close[dayIndex]
            dataPieces.append(db)

        dataPieces.reverse()
        dates.reverse()
        open.reverse()
        self.dataPoints = dataPieces;
        self.dates = dates
        self.open = open

    def run(self):

        s = p.Series(self.open, self.dates)
        s.cumsum()


        movAvg100 = s.resample("1D").fillna('ffill').rolling(window=100,)


        print(self.dataPoints[50].getRollingDataForDay(5))
        print(len(self.dates))



        s.plot(style ='k', markersize=5)
        # movAvg5.mean().plot(style='b')
        # movAvg5.min().plot(style='g')
        # movAvg20.max().plot(style='orange')
        movAvg100.mean().plot(style='r')
        # movAvg80.min().plot(style='g')

        initialBalance = 20000
        balance = initialBalance
        multipler = 1
        equity=0
        stockOwned = 0
        onlyBuyWhenNumIsAbove = 0
        onlySellWhenNumIsAbove = 0

        sll = PreProcessor.StopLossLim(0)

        lastHigh = 0

        priceSoldAt = 0

        for dayIndex in range(len(self.dataPoints)):
            db = self.dataPoints[dayIndex]

            # Adjust StopLossLim
            if (stockOwned > 0):
                if (db.open > lastHigh):
                    sll.price = db.open * .90
                    lastHigh = db.open



            # Double peak analysis,
            # if price just peaked and dropped


            # see likelihood that it will go up again,
            # if it seems likely to go up,
            #  buy some stocks,
            # if it seems unlikely dont.
            #  Put a stop limit at some percent loss below buy price.
            #  If reaches some threshold compared to peak,
            # put stop limit,
            # determine risk of not selling for each time amount held on for,
            # then sell when risk outweighs reward



            # Does nothing right now was just curious how volatility affects the market
            recentVolatily = db.getRollingDataForDay(40, 'var')
            longTermVolatily = db.getRollingDataForDay(100, 'var')

            # if (not math.isnan(longTermVolatily)):
            #     if recentVolatily*.9 > longTermVolatily:
            #      plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=10)

            # peaked = db.peaked(60, dataPieces, minWindow=5, maxWindow=40)
            # if peaked:
            #     plt.plot(db.date, db.open, marker='*', linestyle='--', color='pink', markersize=10)

            plt.plot(db.date, sll.price, marker='o', linestyle='--', color='pink', markersize=5)

            # Within range to 20 day min, and its starting to pick up
            goodBuyShortTerm = db.open*.98 <= db.getRollingDataForDay(20, 'min') and db.getRollingDataForDay(50) > db.getRollingDataForDay(100)

            # Within range to 100 day min, and its starting to pick up
            goodBuyLongTerm = db.open*.98 <= db.getRollingDataForDay(100, 'min') and db.getRollingDataForDay(100) > db.getRollingDataForDay(150)

            # Price is well below usual
            wellBelowUsual = (db.open * .97 < db.getRollingDataForDay(100, 'min') and db.getRollingDataForDay(100) > db.getRollingDataForDay(150))\
                             or db.open*.98 < db.getRollingDataForDay(150, 'min')

            # Most expensive its been in 20 days, and its slowing down
            goodSellShortTerm = db.open*.99 >= db.getRollingDataForDay(20, 'max') and db.getRollingDataForDay(5) < db.getRollingDataForDay(50)

            # Most expensive its been in 50 days, and its slowing down
            goodSellLongTerm = db.open*.98 >= db.getRollingDataForDay(50, 'max') and db.getRollingDataForDay(10) < db.getRollingDataForDay(100)


            # belowMarks = db.open <= db.getRollingData(60) #and db.getRollingData(3) < db.getRollingData(10)

            # Buy
            if (goodBuyShortTerm or goodBuyLongTerm or wellBelowUsual) and not (db.open < sll.price) or db.open < priceSoldAt*.98:
                if (balance > db.open):
                    stockToBuy = 1

                    # A number that correlates with how strong the buy will be
                    num = ((db.getRollingDataForDay(50) / db.getRollingDataForDay(4)) - 1) * 100
                    if (num > 1):
                        stockToBuy = int(num)
                    if (num < onlyBuyWhenNumIsAbove):
                        continue
                    plt.plot(db.date, db.open, marker='o', linestyle='--', color='r', markersize=stockToBuy*2)
                    stockToBuy*=multipler
                    print('buy', stockToBuy, 'at', db.open, db.date)
                    if (stockOwned == 0):
                        sll.price = db.open * .90
                    lastHigh = db.open
                    stockOwned+=stockToBuy
                    balance-=stockToBuy*db.open


                    priceSoldAt = sll.price
                else:
                    print('No moneys')
            # Sell
            elif (goodSellLongTerm or goodSellShortTerm) and not (db.open < sll.price):
                if stockOwned > 0:
                    stockToSell = 1

                    # A number that correlates with how strong the sell will be
                    num = (db.getRollingDataForDay(15) / db.getRollingDataForDay(50) - 1) * 100
                    if (num > 1):
                        stockToSell = int(num)
                    if (num < onlySellWhenNumIsAbove):
                        continue
                    if (stockToSell*multipler > stockOwned):
                        stockToSell = (stockOwned)%multipler + 1

                    plt.plot(db.date, db.open, marker='o', linestyle='--', color='b', markersize=stockToSell*5)
                    stockToSell*=multipler
                    print('sell', stockToSell, 'at', db.open, db.date)
                    stockOwned -= stockToSell
                    balance += db.open*stockToSell
            # Stop limit reached, sell everything
            elif(db.open <= sll.price):
                print("Reached Limit Selling ", stockOwned, "on", db.date)
                sellAmt = stockOwned * sll.price
                balance += sellAmt
                stockOwned = 0
                if self.peakedRecently(db, self.dataPoints, dayIndex):
                    plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=30)
                else:
                    plt.plot(db.date, db.open, marker='*', linestyle='--', color='g', markersize=30)

                sll.price = 0
            else:
                    print("no stock")

        currentPrice = self.dataPoints[len(self.dataPoints)-1].open
        stockMoney=stockOwned*currentPrice
        equity = balance + stockMoney
        profit = equity - initialBalance
        percent = (100 - abs(((equity/initialBalance)*100)))*-1
        print('profit:',profit)
        print('stock money:',stockMoney)
        print('equity:',equity)
        print('gain percent:', percent)

        print('balance', balance)

        print(stockOwned)
        # plt.plot(dates, movAvg80, "b")
        plt.ylabel('some numbers')
        plt.show()

    def movingAvgIndex(self, db):
        # Within range to 20 day min, and its starting to pick up
        goodBuyShortTerm = db.open * .98 <= db.getRollingDataForDay(20, 'min') and db.getRollingDataForDay(
            50) > db.getRollingDataForDay(100)

        # Within range to 100 day min, and its starting to pick up
        goodBuyLongTerm = db.open * .98 <= db.getRollingDataForDay(100, 'min') and db.getRollingDataForDay(
            100) > db.getRollingDataForDay(150)

        # Price is well below usual
        wellBelowUsual = (db.open * .97 < db.getRollingDataForDay(100, 'min') and db.getRollingDataForDay(
            100) > db.getRollingDataForDay(150)) \
                         or db.open * .98 < db.getRollingDataForDay(150, 'min')

        # Most expensive its been in 20 days, and its slowing down
        goodSellShortTerm = db.open * .99 >= db.getRollingDataForDay(20, 'max') and db.getRollingDataForDay(
            5) < db.getRollingDataForDay(50)

        # Most expensive its been in 50 days, and its slowing down
        goodSellLongTerm = db.open * .98 >= db.getRollingDataForDay(50, 'max') and db.getRollingDataForDay(
            10) < db.getRollingDataForDay(100)

        # belowMarks = db.open <= db.getRollingData(60) #and db.getRollingData(3) < db.getRollingData(10)

        num = 0
        # Buy
        if goodBuyShortTerm or goodBuyLongTerm or wellBelowUsual:
            # A number that correlates with how strong the buy will be
            num = ((db.getRollingDataForDay(50) / db.getRollingDataForDay(4)) - 1) * 100
        # Sell
        elif goodSellLongTerm or goodSellShortTerm:
            # A number that correlates with how strong the sell will be
            num = -1*(db.getRollingDataForDay(15) / db.getRollingDataForDay(50) - 1) * 100
        return num

    def diff_from_moving_avg(self, db, period):
        movAvg = db.getRollingDataForDay(period)
        return db.open/movAvg


    # Using this to get best markers for certain things
    def bare_run (self):

        # Plot actual open data
        s = p.Series(self.open, self.dates)
        s.cumsum()
        s.plot(style ='k', markersize=5)

        for dayIndex in range(len(self.dataPoints)):
            db = self.dataPoints[dayIndex]
            if (dayIndex > 5):
                slope = self.calculateSlope(5, db.getRollingData(5), db.date)
                print(slope)
            num = self.movingAvgIndex(db)
            # if num > 0:
            #     plt.plot(db.date, db.open, marker='o', linestyle='--', color='r', markersize=num)
            # elif num < 0:
            #     plt.plot(db.date, db.open, marker='o', linestyle='--', color='b', markersize=num*-1)
            if db.peaked(150, self.dataPoints, minWindow=40, maxWindow=100):
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=4)
            if db.peaked(250, self.dataPoints, minWindow=40, maxWindow=50):
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='y', markersize=4)
            if db.peaked(30, self.dataPoints, minWindow=10, maxWindow=20, MinToMaxRatio=.94):
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='r', markersize=4)
            if db.peaked(80, self.dataPoints):
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=8)

        plt.show()


    def csv_indices(self):
        columns = ['open','percent300Avg', 'growth5', 'growth10', 'growth20','growth40','growth80','growth160','growth300',
                   'num', 'x3', 'x4', 'x5', 'slope5', 'slope10', 'slope20', 'slope40', 'slope80',
                   'slope160','movAvgDiff20','movAvgDiff40', 'movAvgDiff80','movAvgDiff160','var20','var40','var80','var160',
                   'max5D', 'max10D', 'max25D', 'max50', 'max100', 'min5D', 'min10D', 'min25D', 'min50', 'min100',
                   'sp_slope5', 'sp_slope10', 'sp_slope20', 'sp_slope40', 'sp_slope80', 'sp_slope160','sp_movAvgDiff20','sp_movAvgDiff40',
                   'sp_movAvgDiff80','sp_movAvgDiff160','sp_var20','sp_var40','sp_var80','sp_var160',
                   'dow_slope5', 'dow_slope10', 'dow_slope20', 'dow_slope40', 'dow_slope80', 'dow_slope160',
                   'dow_movAvgDiff20', 'dow_movAvgDiff40',
                   'dow_movAvgDiff80', 'dow_movAvgDiff160', 'dow_var20', 'dow_var40', 'dow_var80', 'dow_var160',
                   '5DayActual','10DayActual','50DayActual','k']
        index = self.dates
        df = p.DataFrame(index=index, columns=columns)

        # TODO make yahoo stock reader.. Google has disapointed me, and does not offer csv's off indices
        dowData = PreProcessor("DOW", self.start, self.end, source='Yahoo', local=True)
        spData = PreProcessor("SP", self.start, self.end, source='Yahoo', local=True)

        # TODO add DOW && S&P index
        # TODO add PercentDiffBetweenMovAvg
        # TODO add PE ratio

        # TODO add  5, 10, 20, 40, 80, 160  days later actual db.open


        for dayIndex in range(300, len(self.dataPoints)):
            db = self.dataPoints[dayIndex]
            df.at[db.date, 'open'] = db.open

            df.at[db.date, 'percent300Avg'] = db.getRollingDataForDay(300)/db.open

            if dayIndex>300:
                df.at[db.date, 'growth5'] = db.open/self.open[dayIndex-5]
                df.at[db.date, 'growth10'] = db.open/self.open[dayIndex-10]
                df.at[db.date, 'growth20'] = db.open/self.open[dayIndex-20]
                df.at[db.date, 'growth40'] = db.open/self.open[dayIndex-40]
                df.at[db.date, 'growth80'] = db.open/self.open[dayIndex-80]
                df.at[db.date, 'growth160'] = db.open/self.open[dayIndex-160]
                df.at[db.date, 'growth300'] = db.open/self.open[dayIndex-300]

            num = self.movingAvgIndex(db)
            if num > 0:
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='r', markersize=num)
            elif num < 0:
                plt.plot(db.date, db.open, marker='o', linestyle='--', color='b', markersize=num*-1)
            df.at[db.date, 'num'] = num
            if db.peaked(150, self.dataPoints, minWindow=40, maxWindow=100):
                df.at[db.date, 'x3'] = 1
            else:
                df.at[db.date, 'x3'] = 0
            if db.peaked(250, self.dataPoints, minWindow=40, maxWindow=50):
                df.at[db.date, 'x4'] = 1
            else:
                df.at[db.date, 'x4'] = 0
            if db.peaked(30, self.dataPoints, minWindow=10, maxWindow=20, MinToMaxRatio=.94):
                df.at[db.date, 'x5'] = 1
            else:
                df.at[db.date, 'x5'] = 0
            if (dayIndex >= 5):
                df.at[db.date, 'slope5'] = self.calculateSlope(5, db.getRollingData(5), db.date)
            if (dayIndex >= 10):
                 df.at[db.date, 'slope10'] = self.calculateSlope(10, db.getRollingData(10), db.date)
            if (dayIndex >= 20):
                df.at[db.date, 'slope20'] = self.calculateSlope(20, db.getRollingData(20), db.date)
            if (dayIndex >= 40):
                df.at[db.date, 'slope40'] = self.calculateSlope(40, db.getRollingData(40), db.date)
            if (dayIndex >= 80):
                df.at[db.date, 'slope80'] = self.calculateSlope(80, db.getRollingData(80), db.date)
            if (dayIndex >= 160):
                df.at[db.date, 'slope160'] = self.calculateSlope(160, db.getRollingData(160), db.date)
            if (dayIndex >= 20):
                df.at[db.date, 'movAvgDiff20'] = self.diff_from_moving_avg(db, 20)
            if (dayIndex >= 40):
                df.at[db.date, 'movAvgDiff40'] = self.diff_from_moving_avg(db, 40)
            if (dayIndex >= 80):
                df.at[db.date, 'movAvgDiff80'] = self.diff_from_moving_avg(db, 80)
            if (dayIndex >= 160):
                df.at[db.date, 'movAvgDiff160'] = self.diff_from_moving_avg(db, 160)
            if (dayIndex >= 20):
                df.at[db.date, 'var20'] = db.getRollingDataForDay(20,operation='var')
            if (dayIndex >= 40):
                df.at[db.date, 'var40'] = db.getRollingDataForDay(40,operation='var')
            if (dayIndex >= 80):
                df.at[db.date, 'var80'] = db.getRollingDataForDay(80,operation='var')
            if (dayIndex >= 160):
                df.at[db.date, 'var160'] = db.getRollingDataForDay(160,operation='var')
            if (dayIndex >= 300):
                df.at[db.date, '300MovAvg'] = db.getRollingDataForDay(300)
            df.at[db.date, 'max5D'] = db.open/db.getRollingDataForDay(5, 'max')
            df.at[db.date, 'max10D'] = db.open/db.getRollingDataForDay(10, 'max')
            df.at[db.date, 'max25D'] = db.open/db.getRollingDataForDay(25, 'max')
            df.at[db.date, 'max50D'] = db.open/db.getRollingDataForDay(50, 'max')

            df.at[db.date, 'min5D'] = db.open/db.getRollingDataForDay(5, 'min')
            df.at[db.date, 'min10D'] = db.open/db.getRollingDataForDay(10, 'min')
            df.at[db.date, 'min25D'] = db.open/db.getRollingDataForDay(25, 'min')
            df.at[db.date, 'min50D'] = db.open/db.getRollingDataForDay(50, 'min')

            if dayIndex + 5 < len(self.open):
                df.at[db.date, '5DayActual'] = self.open[dayIndex + 5]/db.open
            if dayIndex + 10 < len(self.open):
                df.at[db.date, '10DayActual'] = self.open[dayIndex + 10]/db.open
            if dayIndex + 50 < len(self.open):
                df.at[db.date, '50DayActual'] = self.open[dayIndex + 50]/db.open

        # TODO this should only be computed once... but I'm lazy
        for dayIndex in range(300, len(spData.dataPoints)):
            db = spData.dataPoints[dayIndex]
            if (dayIndex >= 5):
                df.at[db.date, 'sp_slope5'] = self.calculateSlope(5, db.getRollingData(5), db.date)
            if (dayIndex >= 10):
                 df.at[db.date, 'sp_slope10'] = self.calculateSlope(10, db.getRollingData(10), db.date)
            if (dayIndex >= 20):
                df.at[db.date, 'sp_slope20'] = self.calculateSlope(20, db.getRollingData(20), db.date)
            if (dayIndex >= 40):
                df.at[db.date, 'sp_slope40'] = self.calculateSlope(40, db.getRollingData(40), db.date)
            if (dayIndex >= 80):
                df.at[db.date, 'sp_slope80'] = self.calculateSlope(80, db.getRollingData(80), db.date)
            if (dayIndex >= 160):
                df.at[db.date, 'sp_slope160'] = self.calculateSlope(160, db.getRollingData(160), db.date)
            if (dayIndex >= 20):
                df.at[db.date, 'sp_movAvgDiff20'] = self.diff_from_moving_avg(db, 20)
            if (dayIndex >= 40):
                df.at[db.date, 'sp_movAvgDiff40'] = self.diff_from_moving_avg(db, 40)
            if (dayIndex >= 80):
                df.at[db.date, 'sp_movAvgDiff80'] = self.diff_from_moving_avg(db, 80)
            if (dayIndex >= 160):
                df.at[db.date, 'sp_movAvgDiff160'] = self.diff_from_moving_avg(db, 160)
            if (dayIndex >= 20):
                df.at[db.date, 'sp_var20'] = db.getRollingDataForDay(20,operation='var')
            if (dayIndex >= 40):
                df.at[db.date, 'sp_var40'] = db.getRollingDataForDay(40,operation='var')
            if (dayIndex >= 80):
                df.at[db.date, 'sp_var80'] = db.getRollingDataForDay(80,operation='var')
            if (dayIndex >= 160):
                df.at[db.date, 'sp_var160'] = db.getRollingDataForDay(160,operation='var')

        for dayIndex in range(300, len(dowData.dataPoints)):
            db = dowData.dataPoints[dayIndex]
            if (dayIndex >= 5):
                df.at[db.date, 'dow_slope5'] = self.calculateSlope(5, db.getRollingData(5), db.date)
            if (dayIndex >= 10):
                df.at[db.date, 'dow_slope10'] = self.calculateSlope(10, db.getRollingData(10), db.date)
            if (dayIndex >= 20):
                df.at[db.date, 'dow_slope20'] = self.calculateSlope(20, db.getRollingData(20), db.date)
            if (dayIndex >= 40):
                df.at[db.date, 'dow_slope40'] = self.calculateSlope(40, db.getRollingData(40), db.date)
            if (dayIndex >= 80):
                df.at[db.date, 'dow_slope80'] = self.calculateSlope(80, db.getRollingData(80), db.date)
            if (dayIndex >= 160):
                df.at[db.date, 'dow_slope160'] = self.calculateSlope(160, db.getRollingData(160), db.date)
            if (dayIndex >= 20):
                df.at[db.date, 'dow_movAvgDiff20'] = self.diff_from_moving_avg(db, 20)
            if (dayIndex >= 40):
                df.at[db.date, 'dow_movAvgDiff40'] = self.diff_from_moving_avg(db, 40)
            if (dayIndex >= 80):
                df.at[db.date, 'dow_movAvgDiff80'] = self.diff_from_moving_avg(db, 80)
            if (dayIndex >= 160):
                df.at[db.date, 'dow_movAvgDiff160'] = self.diff_from_moving_avg(db, 160)
            if (dayIndex >= 20):
                df.at[db.date, 'dow_var20'] = db.getRollingDataForDay(20, operation='var')
            if (dayIndex >= 40):
                df.at[db.date, 'dow_var40'] = db.getRollingDataForDay(40, operation='var')
            if (dayIndex >= 80):
                df.at[db.date, 'dow_var80'] = db.getRollingDataForDay(80, operation='var')
            if (dayIndex >= 160):
                df.at[db.date, 'dow_var160'] = db.getRollingDataForDay(160, operation='var')

            # if db.peaked(80, self.dataPoints):
            #     plt.plot(db.date, db.open, marker='o', linestyle='--', color='g', markersize=8)
        df.to_csv("Processed_csv\\" + self.company + "_processed.csv")
    def peakedRecently(self, db, index, peakLookBack=3):

        if index >= peakLookBack and db.getRollingDataForDay(2, 'mean') > db.getRollingDataForDay(5, 'mean'):
            for indexOfRecentDays in range(0, peakLookBack):
                if self.dataPoints[index - indexOfRecentDays].peaked(80, self.dataPoints, minWindow=20, maxWindow=40):
                    return True
        return False
    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def calculateSlope(self, numDays, series, date):
        xs = []
        ys = []
        for i in range(0, numDays):
            newDate = date - datetime.timedelta(days=i)
            ys.append(series[newDate])
            xs.append(i)

        ys.reverse()
        from scipy.stats import linregress
        return linregress(xs, ys)[0]


if __name__ == "__main__":
    StockReader.getStockUrlWSJ("DIJA", start=datetime.date(2011,1,1), end=datetime.date.today())
