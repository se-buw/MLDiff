import unittest
from z3 import *
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
import mldiff.logReg2smt as logReg2smt
import mldiff.mlp2smt as mlp2smt

from tests.helper import TestHelper


class TestClf2SmtIris(unittest.TestCase):

    @unittest.skip
    def test_iris_dt(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        classVar = Int("class")
        s = dt2smt.toSMT(clf, str(classVar), Solver())

        TestHelper.checkClf(self, X, clf, s, classVar, "DT - Iris")

    @unittest.skip
    def test_iris_svm_argMaxSquare(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = LinearSVC(max_iter=5000)
        clf.fit(X, y)
        classVar = Int("class")
        s = svm2smt.toSMT(clf, str(classVar), Solver(), "argMaxSquare")

        TestHelper.checkClf(self, X, clf, s, classVar, "SVM - argMaxSquare - Iris")

    @unittest.skip
    def test_iris_svm_argMaxImpl(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = LinearSVC(max_iter=5000)
        clf.fit(X, y)
        classVar = Int("class")
        s = svm2smt.toSMT(clf, str(classVar), Solver(), "argMaxImpl")

        TestHelper.checkClf(self, X, clf, s, classVar, "SVM - argMaxImpl - Iris")

    @unittest.skip
    def test_iris_logReg_argMaxSquare(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, y)
        classVar = Int("class")
        s = logReg2smt.toSMT(clf, str(classVar), Solver(), "argMaxSquare")
        # print(s)

        TestHelper.checkClf(self, X, clf, s, classVar, "LogReg - argMaxSquare - Iris")

    @unittest.skip
    def test_iris_logReg_argMaxImpl(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, y)
        classVar = Int("class")
        s = logReg2smt.toSMT(clf, str(classVar), Solver(), "argMaxImpl")
        # print(s)

        TestHelper.checkClf(self, X, clf, s, classVar, "LogReg - argMaxImpl - Iris")

    @unittest.skip
    def test_iris_mlp_1_argMaxSquare(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = MLPClassifier(max_iter=2000, hidden_layer_sizes=(1,))
        clf.fit(X, y)
        classVar = Int("class")
        s = mlp2smt.toSMT(clf, str(classVar), Solver(), "argMaxSquare")

        TestHelper.checkClf(
            self, X, clf, s, classVar, "MLP - HL1 - N1 - argMaxSquare - Iris"
        )

    @unittest.skip
    def test_iris_mlp_1_argMaxImpl(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = MLPClassifier(max_iter=2000, hidden_layer_sizes=(1,))
        clf.fit(X, y)
        classVar = Int("class")
        s = mlp2smt.toSMT(clf, str(classVar), Solver(), "argMaxImpl")

        TestHelper.checkClf(
            self, X, clf, s, classVar, "MLP - HL1 - N1 - argMaxImpl - Iris"
        )

    @unittest.skip
    def test_iris_mlp_2_argMaxSquare(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = MLPClassifier(max_iter=2000, hidden_layer_sizes=(2,))
        clf.fit(X, y)
        classVar = Int("class")
        s = mlp2smt.toSMT(clf, str(classVar), Solver(), "argMaxSquare")

        TestHelper.checkClf(
            self, X, clf, s, classVar, "MLP - HL1 - N2 - argMaxSquare - Iris"
        )

    @unittest.skip
    def test_iris_mlp_2_argMaxImpl(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        clf = MLPClassifier(max_iter=2000, hidden_layer_sizes=(2,))
        clf.fit(X, y)
        classVar = Int("class")
        s = mlp2smt.toSMT(clf, str(classVar), Solver(), "argMaxImpl")

        TestHelper.checkClf(
            self, X, clf, s, classVar, "MLP - HL1 - N2 - argMaxImpl - Iris"
        )

    # @unittest.skip
    def test_iris_exhaust(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        clf2 = LinearSVC()
        clf2.fit(X, y)
        print(clf2.coef_)
        print(clf2.intercept_)

        classVar2 = Int("class")
        s = svm2smt.toSMT(clf2, str(classVar2))

        TestHelper.checkAllClasses(self, clf2, s, classVar2, "SVC Iris")

    @unittest.skip
    def test_iris_dt_exhaust(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        clf2 = DecisionTreeClassifier()
        clf2.fit(X, y)

        classVar2 = Int("class")
        s = dt2smt.toSMT(clf2, str(classVar2))

        TestHelper.checkAllClasses(self, clf2, s, classVar2, "DT Iris")


if __name__ == "__main__":
    unittest.main()
