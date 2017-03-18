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
                        0)
    transpose = zip(*fs.yVectors)
    for yVector in transpose:
        lr = LinearRegression()
        # rlasso.fit(fs.xVectors, yVector)
        rfe = RFE(lr, n_features_to_select=1)
        rfe.fit(fs.xVectors, yVector)
        print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), xCols), reverse=False))
