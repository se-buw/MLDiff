from z3 import *
from sklearn.svm import LinearSVC

from mldiff.helper import argMaxImpl as argMaxImpl
from mldiff.helper import argMaxSquare as argMaxSquare
from mldiff.helper import dotProduct as dotProduct


def toSMT(
    svc: LinearSVC,
    clName: str = "class",
    solver: Solver = Solver(),
    argMax: str = "argMaxSquare",
):

    print("Support Vector Machine to SMT\n")

    numClasses: int = len(svc.classes_)
    features: list = []
    for i in range(svc.n_features_in_):
        features.append(Real("x" + str(i)))
    s = solver
    # variable for class of prediction
    cls = Int(clName)
    s.add(cls >= 0, cls < numClasses)

    scores = dotProduct(svc, numClasses, features)

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
