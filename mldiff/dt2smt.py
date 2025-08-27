from z3 import *
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def toSMT(dt: DecisionTreeClassifier, clName: str = "class", solver: Solver = Solver()):
    """
    Convert a decision tree to a formula in Z3 using the features and the class variable.
    The formula is added to the solver. The solver is returned.

    """

    print("Decision Tree to SMT\n")

    formulas: dict = {}
    features: dict = {}

    s = None

    for i in range(dt.n_classes_):
        formulas.setdefault(i, [])
    for i in range(dt.n_features_in_):
        features.setdefault(i, Real("x" + str(i)))
    s = solver

    # variable for class of prediction
    cls = Int(clName)
    s.add(cls >= 0, cls < dt.n_classes_.item())
    formulaForNode(dt, True, 0, formulas, features)
    for i in range(dt.n_classes_):
        if len(formulas[i]) > 0:
            if len(formulas[i]) == 1:
                s.add(formulas[i][0] == (cls == i))
            else:
                s.add(Or(formulas[i]) == (cls == i))
    return s


def formulaForNode(
    dt: DecisionTreeClassifier,
    formula: BoolRef,
    node_id: int,
    formulas: dict,
    features: dict,
):
    """Recursively build the formulas for the paths from root to leafs in the decision tree.
    The formulas are stored in the formulas dictionary accessed by their predicted class.
    """
    is_split_node = dt.tree_.children_left[node_id] != dt.tree_.children_right[node_id]
    if not is_split_node:
        classNum = np.argmax(dt.tree_.value[node_id])
        formulas[classNum].append(formula)
    else:
        fNum = dt.tree_.feature[node_id]
        tVal = dt.tree_.threshold[node_id]
        f = features[fNum]
        if formula == True:
            right = f > tVal
            left = f <= tVal
        else:
            right = And(formula, f > tVal)
            left = And(formula, f <= tVal)
        formulaForNode(dt, right, dt.tree_.children_right[node_id], formulas, features)
        formulaForNode(dt, left, dt.tree_.children_left[node_id], formulas, features)
