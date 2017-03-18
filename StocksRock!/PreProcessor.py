import pandas
import datetime
from dateutil import parser

from StockUrl import StockUrl

from pytrends.request import TrendReq


class PreProcessor:
    def __init__(self, company, searchTerms, startDate, endDate, source):
        if (source == 'Google'):
            url = StockUrl.getStockUrlGoogle(company, startDate, endDate)
        elif (source == 'Yahoo'):
            url = StockUrl.getStockUrlYahoo(company, startDate, endDate)
        else:
            raise ValueError("Invalid source")
        self.data = pandas.read_csv(url, parse_dates=True, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
                                    header=0)
        self.company = company
        self.startDate = startDate
        self.endDate = endDate
        self.terms = searchTerms

    def getData(self):
        self.dates = [parser.parse(x) for x in self.data['Date'].values]
        self.opens = [float(x) for x in self.data['Open'].values]
        self.highs = [float(x) for x in self.data['High'].values]
        self.lows = [float(x) for x in self.data['Low'].values]
        self.closes = [float(x) for x in self.data['Close'].values]

        self.dates.reverse()
        self.opens.reverse()
        self.highs.reverse()
        self.lows.reverse()
        self.closes.reverse()



        self.getTrendData(self.terms)

    def createCSV(self):
        columns = ['dayVec' + str(2 ** x) for x in range(0, 9)]
        df = pandas.DataFrame(index=self.dates, columns=columns)
        # Difference in closing times
        dayVec1 = self.createVectorForDateRange(1)
        dayVec2 = self.createVectorForDateRange(2)
        dayVec4 = self.createVectorForDateRange(4)
        dayVec8 = self.createVectorForDateRange(8)
        dayVec16 = self.createVectorForDateRange(16)
        dayVec32 = self.createVectorForDateRange(32)
        dayVec64 = self.createVectorForDateRange(64)
        dayVec128 = self.createVectorForDateRange(128)
        dayVec256 = self.createVectorForDateRange(256)

        dayVec1Forward = self.createVectorForDateRangeForward(1)
        dayVec2Forward = self.createVectorForDateRangeForward(2)
        dayVec4Forward = self.createVectorForDateRangeForward(4)
        dayVec8Forward = self.createVectorForDateRangeForward(8)
        dayVec16Forward = self.createVectorForDateRangeForward(16)
        dayVec32Forward = self.createVectorForDateRangeForward(32)
        dayVec64Forward = self.createVectorForDateRangeForward(64)

        s = pandas.Series(self.closes, self.dates)
        s.cumsum()

        # resampled = s.resample("1D").fillna('ffill')
        resampled = s.interpolate(method='polynomial', order=1)
        movAvg8 = resampled.rolling(window=8).mean()
        movAvg16 = resampled.rolling(window=16).mean()
        movAvg32 = resampled.rolling(window=32).mean()
        movAvg64 = resampled.rolling(window=64).mean()
        movAvg128 = resampled.rolling(window=128).mean()
        movAvg256 = resampled.rolling(window=256).mean()

        movMin8 = resampled.rolling(window=8).min()
        movMin16 = resampled.rolling(window=16).min()
        movMin32 = resampled.rolling(window=32).min()
        movMin64 = resampled.rolling(window=64).min()
        movMin128 = resampled.rolling(window=128).min()
        movMin256 = resampled.rolling(window=256).min()

        movMax8 = resampled.rolling(window=8).max()
        movMax16 = resampled.rolling(window=16).max()
        movMax32 = resampled.rolling(window=32).max()
        movMax64 = resampled.rolling(window=64).max()
        movMax128 = resampled.rolling(window=128).max()
        movMax256 = resampled.rolling(window=256).max()

        movMed8 = resampled.rolling(window=8).median()
        movMed16 = resampled.rolling(window=16).median()
        movMed32 = resampled.rolling(window=32).median()
        movMed64 = resampled.rolling(window=64).median()
        movMed128 = resampled.rolling(window=128).median()
        movMed256 = resampled.rolling(window=256).median()

        movSTD8 = resampled.rolling(window=8).std()
        movSTD16 = resampled.rolling(window=16).std()
        movSTD32 = resampled.rolling(window=32).std()
        movSTD64 = resampled.rolling(window=64).std()
        movSTD128 = resampled.rolling(window=128).std()
        movSTD256 = resampled.rolling(window=256).std()

        meanTrendsData = []
        meanTrendsColumns = []
        allTrendSeries = []

        # For each term make a list of all series for window size from 8 to 256(doubling) and corresponding names
        for i, term in enumerate(terms):
            pytrends = self.allPyTrends[i].interest_over_time()
            allTrendSeries.append(pytrends[term].resample("1D").interpolate(method='polynomial', order=1))
            meanTrendsData.append(self.getSeriesData(pytrends, 8, 256, name=term))
            meanTrendsColumns.append(self.getColNames(8, 256, term + "trends"))


        for i, d in enumerate(self.dates):
            # print(trendsDF.at[d, 'stock'])
            df.at[d, 'dayVec1'] = dayVec1[i]
            df.at[d, 'dayVec2'] = dayVec2[i]
            df.at[d, 'dayVec4'] = dayVec4[i]
            df.at[d, 'dayVec8'] = dayVec8[i]
            df.at[d, 'dayVec16'] = dayVec16[i]
            df.at[d, 'dayVec32'] = dayVec32[i]
            df.at[d, 'dayVec64'] = dayVec64[i]
            df.at[d, 'dayVec128'] = dayVec128[i]
            df.at[d, 'dayVec256'] = dayVec256[i]

            df.at[d, 'movAvg8'] = self.closes[i] / movAvg8[d] - 1
            df.at[d, 'movAvg16'] = self.closes[i] / movAvg16[d] - 1
            df.at[d, 'movAvg32'] = self.closes[i] / movAvg32[d] - 1
            df.at[d, 'movAvg64'] = self.closes[i] / movAvg64[d] - 1
            df.at[d, 'movAvg128'] = self.closes[i] / movAvg128[d] - 1
            df.at[d, 'movAvg256'] = self.closes[i] / movAvg256[d] - 1

            df.at[d, 'movMin8'] = self.closes[i] / movMin8[d] - 1
            df.at[d, 'movMin16'] = self.closes[i] / movMin16[d] - 1
            df.at[d, 'movMin32'] = self.closes[i] / movMin32[d] - 1
            df.at[d, 'movMin64'] = self.closes[i] / movMin64[d] - 1
            df.at[d, 'movMin128'] = self.closes[i] / movMin128[d] - 1
            df.at[d, 'movMin256'] = self.closes[i] / movMin256[d] - 1

            df.at[d, 'movMax8'] = self.closes[i] / movMax8[d] - 1
            df.at[d, 'movMax16'] = self.closes[i] / movMax16[d] - 1
            df.at[d, 'movMax32'] = self.closes[i] / movMax32[d] - 1
            df.at[d, 'movMax64'] = self.closes[i] / movMax64[d] - 1
            df.at[d, 'movMax128'] = self.closes[i] / movMax128[d] - 1
            df.at[d, 'movMax256'] = self.closes[i] / movMax256[d] - 1

            df.at[d, 'movMed8'] = self.closes[i] / movMed8[d] - 1
            df.at[d, 'movMed16'] = self.closes[i] / movMed16[d] - 1
            df.at[d, 'movMed32'] = self.closes[i] / movMed32[d] - 1
            df.at[d, 'movMed64'] = self.closes[i] / movMed64[d] - 1
            df.at[d, 'movMed128'] = self.closes[i] / movMed128[d] - 1
            df.at[d, 'movMed256'] = self.closes[i] / movMed256[d] - 1

            df.at[d, 'movSTD8'] = self.closes[i] / movSTD8[d] - 1
            df.at[d, 'movSTD16'] = self.closes[i] / movSTD16[d] - 1
            df.at[d, 'movSTD32'] = self.closes[i] / movSTD32[d] - 1
            df.at[d, 'movSTD64'] = self.closes[i] / movSTD64[d] - 1
            df.at[d, 'movSTD128'] = self.closes[i] / movSTD128[d] - 1
            df.at[d, 'movSTD256'] = self.closes[i] / movSTD256[d] - 1

            # df.at[d, 'movAvgTrends8'] = trendsData[d] / movAvgTrends8[d] - 1
            # df.at[d, 'movAvgTrends16'] = trendsData[d] / movAvgTrends16[d] - 1
            # df.at[d, 'movAvgTrends32'] = trendsData[d] / movAvgTrends32[d] - 1
            # df.at[d, 'movAvgTrends64'] = trendsData[d] / movAvgTrends64[d] - 1
            # df.at[d, 'movAvgTrends128'] = trendsData[d] / movAvgTrends128[d] - 1
            # df.at[d, 'movAvgTrends256'] = trendsData[d] / movAvgTrends256[d] - 1

            for j in range(len(terms)):
                for k in range(len(meanTrendsColumns[j])):
                    df.at[d, meanTrendsColumns[j][k]] = allTrendSeries[j][d] / meanTrendsData[j][k][d]

            df.at[d, 'dayVec1Forward'] = dayVec1Forward[i]
            df.at[d, 'dayVec2Forward'] = dayVec2Forward[i]
            df.at[d, 'dayVec4Forward'] = dayVec4Forward[i]
            df.at[d, 'dayVec8Forward'] = dayVec8Forward[i]
            df.at[d, 'dayVec16Forward'] = dayVec16Forward[i]
            df.at[d, 'dayVec32Forward'] = dayVec32Forward[i]
            df.at[d, 'dayVec64Forward'] = dayVec64Forward[i]

        df.to_csv("processed_csv/" + self.company + ".csv")

    def createVectorForDateRange(self, days):
        vector = [None for x in range(0, days)]
        for i in range(days, len(self.closes)):
            change = self.closes[i] / self.closes[i - days] - 1
            vector.append(change)
        return vector

    def createVectorForDateRangeForward(self, days):
        vector = []
        for i in range(0, len(self.closes) - days):
            change = self.closes[i] / self.closes[i + days] - 1
            vector.append(change)
        vector += [None for x in range(0, days)]
        return vector

    def getTrendData(self, searchTerms):
        self.allPyTrends = []
        # By passing in groups of terms it pits them against eachother biasing them, so instead make individual calls
        for term in searchTerms:
            arr = [term] # need list
            pytrends = TrendReq("stockdatamining@gmail.com", "JeffPhillips", hl='en-US', tz=360, custom_useragent=None)
            pytrends.build_payload(arr, timeframe='all')
            self.allPyTrends.append(pytrends)




    @staticmethod
    def getSeriesData(data, start, end, incrementType='double', method='mean', name=''):
        resampled = None
        if name != '':
            resampled = data[name].resample("1D").interpolate(method='polynomial', order=1)
        else:
            resampled = data.resample("1D").interpolate(method='polynomial', order=1)

        allSeries = []
        i = start
        while i <= end:
            if method == 'mean':
                allSeries.append(resampled.rolling(window=i).mean())
            if incrementType == 'double':
                i *= 2
        return allSeries

    @staticmethod
    def getColNames(start, end, baseName, incrementType='double'):
        allColumns = []
        i = start
        while i <= end:
            allColumns.append(baseName + str(i))
            if incrementType == 'double':
                i *= 2
        return allColumns


if __name__ == "__main__":
    terms = ["Bear Market", "Bull Market", "Motorola", "MSI", "MSI Buy", "When to buy stocks", "When to sell stocks"]
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2017, 3, 1)
    p = PreProcessor('MSI', terms, start, end, 'Google')
    p.getData()
    p.createCSV()
