from z3 import *
import numpy as np


def linearClf2Smt(
    linClf,
    clName: str,
    solver: Solver,
    argMax: str,
    epsilonPerFeature: float,
    epsilonArgMax: float,
) -> Solver:
    numClasses: int = len(linClf.classes_)
    features: list = []
    featurePrimes: list = []

    for i in range(linClf.n_features_in_):
        features.append(Real("x" + str(i)))
        featurePrimes.append(Real("xP" + str(i)))
    s = solver
    # variable for class of prediction
    cls = Int(clName)
    s.add(cls >= 0, cls < numClasses)

    svcSmt = linearClfFormula(linClf, clName, features, argMax, epsilonArgMax)
    s.add(And(svcSmt))

    if epsilonPerFeature != 0:
        svcPrimeSmt = linearClfFormula(
            linClf, clName, featurePrimes, argMax, epsilonArgMax
        )  # FIXME: should epsilonArgMax be 0?
        # forall features, feature difference withing epsilon means that class is the same as predicted by the tree
        changesToFeature = []
        for i in range(len(features)):
            changesToFeature.append(
                And(
                    features[i] - epsilonPerFeature < featurePrimes[i],
                    featurePrimes[i] < features[i] + epsilonPerFeature,
                )
            )
        robustness = ForAll(featurePrimes, Implies(And(changesToFeature), svcPrimeSmt))
        s.add(robustness)
    return s


def linearClfFormula(
    linClf,
    clName: str,
    features: list,
    argMax: str = "argMaxSquare",
    epsilonArgMax: float = 0,
) -> BoolRef:
    numClasses: int = len(linClf.classes_)
    cls = Int(clName)
    conjuncts = []
    scores = dotProduct(linClf, numClasses, features)

    if numClasses == 2:
        # if there are only two classes we can use a simple if-then-else
        conjuncts.append(cls == If(scores[0] > 0, 1, 0))
    else:
        if argMax == "argMaxSquare":
            conjuncts.append(argMaxSquareFormula(scores, cls))

        elif argMax == "argMaxImpl":
            conjuncts.append(argMaxImplFormula(scores, cls, clName))
        else:
            raise Exception("Invalid argMax function name")

    conjuncts.append(addEpsilonArgMax(epsilonArgMax, numClasses, cls, scores))

    return And(conjuncts)


def argMaxImplFormula(scores: list, cls: Int, name: str):
    conjuncts = []
    numClasses = len(scores)

    theMax = Real("max" + str(name))
    conjuncts.append(Or([theMax == scores[i] for i in range(numClasses)]))
    for i in range(numClasses):
        # if score of i is less than max then class is not i
        conjuncts.append(Implies(scores[i] < theMax, cls != i))
        # class is i iff score is max
        conjuncts.append((cls == i) == (scores[i] == theMax))
        # FIXME: we should pick the class with the lowest index in case of ties
    return And(conjuncts)


def argMaxImpl(s: Solver, scores: list, cls: Int, name: str):
    s.add(argMaxImplFormula(scores, cls, name))


def argMaxSquareFormula(scores: list, cls: Int):
    # idea: for each prediction of a class ensure that it is greater than all other predictions
    # in case of ties (equality) prefer the class with the lower index
    conjuncts = []
    numClasses = len(scores)
    for i in range(numClasses):
        greater: list = []
        for j in range(numClasses):
            if i < j:
                # to be class i this must be greater or equal than all lower class indices
                greater.append(scores[i] >= scores[j])
            elif i > j:
                # to be class i this must be greater to all higher class indices
                greater.append(scores[i] > scores[j])
        conjuncts.append((cls == i) == And(greater))
    return And(conjuncts)


def argMaxSquare(s: Solver, scores: list, cls: Int):
    s.add(argMaxSquareFormula(scores, cls))


def addEpsilonArgMax(epsilonArgMax, numClasses, cls, scores):
    """Add a constraint that the chosen class in argMax is larger than all others by epsilonArgMax."""
    conjuncts = []
    if numClasses == 2:
        # if epsilonArgMax is not 0, we add a constraint that the distance of scores[0] to the decision boundary is at least epsilonArgMax to both sides
        if epsilonArgMax != 0:
            conjuncts.append(
                Or(
                    And(cls == 0, scores[0] < -epsilonArgMax),
                    And(cls == 1, scores[0] > epsilonArgMax),
                )
            )
    else:
        # if epsilonArgMax is not 0, we add a constraint that the chosen class is larger than all others by epsilonArgMax
        if epsilonArgMax != 0:
            for i in range(numClasses):
                largerByEpsilon = And(
                    [
                        (scores[i] - epsilonArgMax >= scores[j])
                        for j in range(numClasses)
                        if j != i
                    ]
                )
                conjuncts.append(Implies(cls == i, largerByEpsilon))
    return And(conjuncts)


def dotProduct(classifier, numClasses: int, features) -> list:
    # dot product of features and coefficients plus intercept
    m = classifier.coef_.T.transpose()
    scores: list = []

    # scikit-learn stores the coefficients in a different way for binary and multiclass classification
    # @see:  https://scikit-learn.org/stable/modules/svm.html#details-on-multi-class-strategies
    # binary: 1 x n_features
    # multiclass: n_classes x n_features
    # we need to handle both cases
    if numClasses == 2 and m.shape[0] == 1:
        assert len(features) == len(m[0])
        pairs = zip(features, m[0])
        # multiply each pair and sum the results to get the dot product, then add the intercept
        scores.append(
            Sum(list(map(lambda pair: pair[0] * RealVal(pair[1]), pairs)))
            + classifier.intercept_[0]
        )
    else:
        for i in range(numClasses):
            assert len(features) == len(m[i])
            # zip each feature in a pair with its coefficients
            pairs = zip(features, m[i])
            # multiply each pair and sum the results to get the dot product, then add the intercept
            scores.append(
                Sum(list(map(lambda pair: pair[0] * RealVal(pair[1]), pairs)))
                + classifier.intercept_[i]
            )

    return scores


def dotProductMLP(classifier, scores, activation, m, i):
    for j in range(len(m)):  # likely go over length or width of m
        # zip each previous activation variable in a pair with its coefficients
        assert len(activation) == len(m[j])
        pairs = zip(activation, m[j])

        # multiply each pair and sum the results to get the dot product, then add the intercept
        scores.append(
            Sum(list(map(lambda pair: pair[0] * RealVal(pair[1]), pairs)))
            + classifier.intercepts_[i][j]
        )

    return scores


def applyActivationFunctionMLP(classifier, scores):
    print("activation function " + classifier.activation)

    if classifier.activation == "identity":
        pass
    if classifier.activation == "relu":
        # map wendet If auf jedes Element von scores an
        scores = list(map(lambda c: If(c > 0, c, 0), scores))
    if classifier.activation == "softmax":
        raise Exception("Softmax activation function not supported")
    if classifier.activation == "tanh":
        raise Exception("Tanh activation function not supported")
    if classifier.activation == "logistic":
        raise Exception("Logistic activation function not supported")

    return scores
