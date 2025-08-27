from z3 import *
from sklearn.neural_network import MLPClassifier

from mldiff.helper import argMaxImpl as argMaxImpl
from mldiff.helper import argMaxSquare as argMaxSquare
from mldiff.helper import dotProductMLP as dotProductMLP
from mldiff.helper import applyActivationFunctionMLP as applyActivationFunctionMLP


def toSMT(
    mlp: MLPClassifier,
    clName: str = "class",
    solver: Solver = Solver(),
    argMax: str = "argMaxSquare",
):

    print("Multilayer Perceptron to SMT\n")

    numClasses: int = len(mlp.classes_)
    features: list = []
    for i in range(mlp.n_features_in_):
        features.append(Real("x" + str(i)))
    s = solver
    # variable for class of prediction
    cls = Int(clName)
    s.add(cls >= 0, cls < len(mlp.classes_))

    # Initialize first layer
    activation = features

    for i in range(mlp.n_layers_ - 1):

        m = mlp.coefs_[i].T
        scores: list = []

        # dot product of features and coefficients plus intercept
        scores = dotProductMLP(mlp, scores, activation, m, i)

        if i != mlp.n_layers_ - 2:

            # apply activation function
            scores = applyActivationFunctionMLP(mlp, scores)

        activation = scores

    if (
        mlp.out_activation_ == "logistic"
        or mlp.out_activation_ == "softmax"
        or mlp.out_activation_ == "identity"
    ):
        if numClasses == 2:
            s.add(cls == If(scores[0] > 0, 1, 0))
        else:
            if argMax == "argMaxSquare":
                argMaxSquare(s, scores, cls)

            if argMax == "argMaxImpl":
                argMaxImpl(s, scores, cls, clName)
    else:
        raise Exception("Invalid output activation function.")

    return s
