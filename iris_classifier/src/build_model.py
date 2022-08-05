#!/usr/bin/env python
"""
Train and build model joblib file
"""

__version__ = "1.0.0"

import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from joblib import dump


def main():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = linear_model.LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    print('Mean accuracy score: %.2f' % clf.score(X_test, y_test))

    dump(clf, '../model/model.joblib')


if __name__ == "__main__":
    main()
