import pandas
import datetime
from dateutil import parser

from StockUrl import StockUrl

class PreProcessor:

    def __init__(self, company, startDate, endDate, source):
        if (source == 'Google'):
            url = StockUrl.getStockUrlGoogle(company, startDate, endDate)
        elif (source == 'Yahoo'):
            url = StockUrl.getStockUrlYahoo(company, startDate, endDate)
        else:
            raise ValueError("Invalid source")
        self.data = pandas.read_csv(url, parse_dates=True, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], header=0)

    def getData(self):
        self.dates  = [parser.parse(x) for x in self.data['Date'].values]
        self.opens  = [float(x) for x in self.data['Open'].values]
        self.highs  = [float(x) for x in self.data['High'].values]
        self.lows   = [float(x) for x in self.data['Low'].values]
        self.closes = [float(x) for x in self.data['Close'].values]

        self.dates.reverse()
        self.opens.reverse()
        self.highs.reverse()
        self.lows.reverse()
        self.closes.reverse()

    def createCSV(self):
        columns=['dayVec' + str(2**x) for x in range(0, 9)]
        df = pandas.DataFrame(index=self.dates, columns=columns)
        # Difference in closing times
        dayVec1   = self.createVectorForDateRange(1)
        dayVec2   = self.createVectorForDateRange(2)
        dayVec4   = self.createVectorForDateRange(4)
        dayVec8   = self.createVectorForDateRange(8)
        dayVec16  = self.createVectorForDateRange(16)
        dayVec32  = self.createVectorForDateRange(32)
        dayVec64  = self.createVectorForDateRange(64)
        dayVec128 = self.createVectorForDateRange(128)
        dayVec256 = self.createVectorForDateRange(256)

        for i, d in enumerate(self.dates):
            df.at[d, 'dayVec1']   = dayVec1[i]
            df.at[d, 'dayVec2']   = dayVec2[i]
            df.at[d, 'dayVec4']   = dayVec4[i]
            df.at[d, 'dayVec8']   = dayVec8[i]
            df.at[d, 'dayVec16']  = dayVec16[i]
            df.at[d, 'dayVec32']  = dayVec32[i]
            df.at[d, 'dayVec64']  = dayVec64[i]
            df.at[d, 'dayVec128'] = dayVec128[i]
            df.at[d, 'dayVec256'] = dayVec256[i]

        df.to_csv("test.csv")

    def createVectorForDateRange(self, days):
        vector = [None for x in range(0, days)]
        for i in range(days, len(self.closes)):
            change = self.closes[i] / self.closes[i-days] - 1
            vector.append(change)
        return vector


if __name__ == "__main__":
    start = datetime.datetime(1990, 1, 1)
    end = datetime.date.today()
    p = PreProcessor('AAPL', start, end, 'Google')
    p.getData()
    p.createCSV()
