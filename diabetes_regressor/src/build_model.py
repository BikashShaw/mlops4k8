#!/usr/bin/env python
"""
Train and build model joblib file
"""

__version__ = "1.0.0"

import numpy as np
from joblib import dump
from sklearn import datasets, linear_model


def main():
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)

    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f" %
          np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2))
    print('Variance score: %.2f' %
          regr.score(diabetes_X_test, diabetes_y_test))

    dump(regr, '../model/model.joblib')


if __name__ == "__main__":
    main()
