import pandas
import numbers
import math
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RandomizedLasso


class FeatureSelection:
    def __init__(self, company, xColumns, yColumns, cvPercent):
        data = pandas.read_csv("processed_csv/" + company + ".csv")
        self.xcols = xColumns
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

if __name__ == "__main__":
    xCols = open('xCols', 'r').read().split(',')
    yCols = open('yCols', 'r').read().split(',')
    rlasso = RandomizedLasso(alpha=0.025)
    fs = FeatureSelection("AAPL", xCols, yCols,
                        .1)
    transpose = zip(*fs.yVectors)

    totalVec = np.zeros(len(xCols))
    for yVector in transpose:
        lr = LinearRegression()
        # rlasso.fit(fs.xVectors, yVector)
        rfe = RFE(lr)
        rfe.fit(fs.xVectors, yVector)
        a = rfe.ranking_
        totalVec += a

    l = (sorted(zip(map(lambda x: round(x, 4), totalVec), xCols), reverse=False))
    for i in range(int(len(l)/2)):
        print(l[i][1] + ",", end="")