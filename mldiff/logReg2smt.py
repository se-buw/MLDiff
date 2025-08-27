from z3 import *
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.datasets import load_iris

from mldiff.helper import argMaxImpl as argMaxImpl
from mldiff.helper import argMaxSquare as argMaxSquare
from mldiff.helper import dotProduct as dotProduct


def toSMT(
    logReg: LogisticRegression,
    clName: str = "class",
    solver: Solver = Solver(),
    argMax: str = "argMaxSquare",
):

    print("Logistic Regression to SMT\n")

    numClasses: int = len(logReg.classes_)
    features: list = []
    for i in range(logReg.n_features_in_):
        features.append(Real("x" + str(i)))
    s = solver
    # variable for class of prediction
    cls = Int(clName)
    s.add(cls >= 0, cls < numClasses)

    scores = dotProduct(logReg, numClasses, features)

    if numClasses == 2:
        # if there are only two classes we can use a simple if-then-else
        s.add(cls == If(scores[0] > 0, 1, 0))
    else:
        if argMax == "argMaxSquare":
            argMaxSquare(s, scores, cls)

        elif argMax == "argMaxImpl":
            argMaxImpl(s, scores, cls, clName)
        else:
            raise Exception("Invalid argMax function name")

    return s


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    logReg = LogisticRegression(max_iter=1000)
    logReg.fit(X, y)

    clLogReg = Int("classLogReg")
    toSMT(logReg, str(clLogReg))


if __name__ == "__main__":
    main()
