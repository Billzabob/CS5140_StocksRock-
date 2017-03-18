import pandas
import numbers
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class LinearAnalyzer:
    def __init__(self, company, xColumns, yColumns, cvPercent):
        data = pandas.read_csv("processed_csv/" + company + ".csv")
        self.xVectors = []
        self.yVectors = []
        self.xVectorsCV = []
        self.yVectorsCV = []
        for i in range(0, len(data)):
            skip = False
            xRow = []
            for x in xColumns:
                val = data[x].values[i]
                if not isinstance(val, numbers.Real) or math.isnan(val):
                    skip = True
                xRow.append(val)
            yRow = []
            for y in yColumns:
                val = data[y].values[i]
                if not isinstance(val, numbers.Real) or math.isnan(val):
                    skip = True
                yRow.append(val)
            if skip:
                continue
            if 1 - (i / len(data)) > cvPercent:
                self.xVectors.append(xRow)
                self.yVectors.append(yRow)
            else:
                self.xVectorsCV.append(xRow)
                self.yVectorsCV.append(yRow)

    def fit(self):
        polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
        linear_regression = LinearRegression()
        self.pipeline = Pipeline(
            [("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
        self.pipeline.fit(self.xVectors, self.yVectors)

    def crossValidate(self):
        guesses4 = []
        actuals4 = []
        guesses64 = []
        actuals64 = []

        for i, r in enumerate(self.xVectorsCV):
            guess = self.pipeline.predict(np.asarray(r).reshape(1, -1))
            guesses4.append(guess[0][2])
            # guesses4.append(0)
            guesses64.append(guess[0][6])
            # guesses64.append(0)
            actual = self.yVectorsCV[i]
            actuals4.append(actual[2])
            actuals64.append(actual[6])

        plt.title('Cross Validation of 4 Day Percent Change')
        plt.xlabel('day')
        plt.ylabel('percent change')
        plt.plot(range(0, len(guesses4)), guesses4, 'r.', label='guess')
        plt.plot(range(0, len(guesses4)), actuals4, 'b.', label='actual')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()

        plt.title('Cross Validation of 64 Day Percent Change')
        plt.xlabel('day')
        plt.ylabel('percent change')
        plt.plot(range(0, len(guesses64)), guesses64, 'r.', label='guess')
        plt.plot(range(0, len(guesses64)), actuals64, 'b.', label='actual')
        plt.show()

        print('4 day squared Error:', self.calculateAvgError(guesses4, actuals4))
        print('4 day absolute Error:', self.calculateAvgError(guesses4, actuals4, method='absolute'))
        print('64 day absoluteError:', self.calculateAvgError(guesses64, actuals64, method='absolute'))
        print('64 day squared Error:', self.calculateAvgError(guesses64, actuals64))

        print('4 day within 1 percent range:', self.cvWithInRange(.01, guesses4, actuals4))
        print('64 day within 5 percent range:', self.cvWithInRange(.05, guesses64, actuals64))

        print('4 day within 2 percent range:', self.cvWithInRange(.02, guesses4, actuals4))
        print('64 day within 10 percent range:', self.cvWithInRange(.1, guesses64, actuals64))

    def calculateAvgError(self, guesses, actuals, method='squared'):
        if len(guesses) != len(actuals):
            raise ValueError()
        avgError = 0
        for i in range(len(guesses)):
            if method == 'squared':
                avgError += (guesses[i] - actuals[i]) ** 2
            elif method == 'absolute':
                avgError += abs(guesses[i] - actuals[i])
            else:
                raise ValueError()
        avgError /= len(guesses)

        return avgError

    # Gives percent of cross validation data that fell within range
    def cvWithInRange(self, maxDif, guesses, actuals):
        if len(guesses) != len(actuals):
            raise ValueError()
        count = 0
        for i in range(len(guesses)):
            if abs(guesses[i] - actuals[i]) <= maxDif:
                count += 1
        return count / len(guesses)


if __name__ == "__main__":
    xCols = open('xCols', 'r').read().split(',')
    yCols = open('yCols', 'r').read().split(',')
    print(xCols)
    print(yCols)

    movMedColumns = ['movMed8', 'movMed16', 'movMed32', 'movMed64', 'movMed128', 'movMed256']
    movStdColumns = ['movSTD8', 'movSTD16', 'movSTD32', 'movSTD64', 'movSTD128', 'movSTD256']
    movMaxColumns = ['movMax8', 'movMax16', 'movMax32', 'movMax64', 'movMax128', 'movMax256']
    movMinColumns = ['movMin8', 'movMin16', 'movMin32', 'movMin64', 'movMin128', 'movMin256']
    movAvgColumns = ['movAvg8', 'movAvg16', 'movAvg32', 'movAvg64', 'movAvg128', 'movAvg256']
    dayVecColumns = ['dayVec1', 'dayVec2', 'dayVec4', 'dayVec8', 'dayVec16', 'dayVec32', 'dayVec64', 'dayVec128',
                     'dayVec256']
    trendColumns = ['movAvgTrends8', 'movAvgTrends16', 'movAvgTrends32', 'movAvgTrends64', 'movAvgTrends128',
                    'movAvgTrends256']

    # lr = LinearAnalyzer("AAPL", dayVecColumns + movAvgColumns + movMinColumns + movMaxColumns + movMedColumns +
    #                     movStdColumns + trendColumns,
    #                     ['dayVec1Forward', 'dayVec2Forward', 'dayVec4Forward', 'dayVec8Forward', 'dayVec16Forward',
    #                      'dayVec32Forward', 'dayVec64Forward'],
    #                     .1)

    lr = LinearAnalyzer("MSI", xCols, yCols,
                        .3)

    lr.fit()
    lr.crossValidate()

# 4 day within 1 percent range: 0.20833333333333334
# 64 day within 5 percent range: 0.25925925925925924
# 4 day within 2 percent range: 0.4722222222222222
# 64 day within 10 percent range: 0.4583333333333333