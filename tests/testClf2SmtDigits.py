import unittest
from z3 import *
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
import mldiff.logReg2smt as logReg2smt
import mldiff.mlp2smt as mlp2smt

from tests.helper import TestHelper


class TestClf2SmtDigits(unittest.TestCase):

    # @unittest.skip
    def test_digits_dt(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = DecisionTreeClassifier(max_depth=10)
        clf.fit(X, y)
        classVar = Int("class")
        s = dt2smt.toSMT(clf, str(classVar), Solver())

        TestHelper.checkClf(self, X, clf, s, classVar, "DT - Digits")

    # @unittest.skip
    def test_digits_svm_argMaxSquare(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = LinearSVC(max_iter=5000)
        clf.fit(X, y)
        classVar = Int("class")
        s = svm2smt.toSMT(clf, str(classVar), Solver(), "argMaxSquare")

        TestHelper.checkClf(self, X, clf, s, classVar, "SVM - argMaxSquare - Digits")

    @unittest.skip
    def test_digits_svm_argMaxImpl(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = LinearSVC(max_iter=5000)
        clf.fit(X, y)
        classVar = Int("class")
        s = svm2smt.toSMT(clf, str(classVar), Solver(), "argMaxImpl")

        TestHelper.checkClf(self, X, clf, s, classVar, "SVM - argMaxImpl - Digits")

    # @unittest.skip
    def test_digits_logReg_argMaxSquare(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, y)
        classVar = Int("class")
        s = logReg2smt.toSMT(clf, str(classVar), Solver(), "argMaxSquare")

        TestHelper.checkClf(self, X, clf, s, classVar, "LogReg - argMaxSquare - Digits")

    @unittest.skip
    def test_digits_logReg_argMaxImpl(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, y)
        classVar = Int("class")
        s = logReg2smt.toSMT(clf, str(classVar), Solver(), "argMaxImpl")

        TestHelper.checkClf(self, X, clf, s, classVar, "LogReg - argMaxImpl - Digits")

    # @unittest.skip
    def test_digits_mlp_1_argMaxSquare(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = MLPClassifier(max_iter=2000, hidden_layer_sizes=(1,))
        clf.fit(X, y)
        classVar = Int("class")
        s = mlp2smt.toSMT(clf, str(classVar), Solver(), "argMaxSquare")

        TestHelper.checkClf(
            self, X, clf, s, classVar, "MLP - HL1 - N1 - argMaxSquare - Digits"
        )

    # @unittest.skip
    def test_digits_mlp_1_argMaxImpl(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = MLPClassifier(max_iter=2000, hidden_layer_sizes=(1,))
        clf.fit(X, y)
        classVar = Int("class")
        s = mlp2smt.toSMT(clf, str(classVar), Solver(), "argMaxImpl")

        TestHelper.checkClf(
            self, X, clf, s, classVar, "MLP - HL1 - N1 - argMaxImpl - Digits"
        )

    @unittest.skip
    def test_digits_mlp_2_argMaxSquare(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = MLPClassifier(max_iter=2000, hidden_layer_sizes=(2,))
        clf.fit(X, y)
        classVar = Int("class")
        s = mlp2smt.toSMT(clf, str(classVar), Solver(), "argMaxSquare")

        TestHelper.checkClf(
            self, X, clf, s, classVar, "MLP - HL1 - N2 - argMaxSquare - Digits"
        )

    @unittest.skip
    def test_digits_mlp_2_argMaxImpl(self):
        digits = load_digits()
        X = digits.data
        y = digits.target

        clf = MLPClassifier(max_iter=2000, hidden_layer_sizes=(2,))
        clf.fit(X, y)
        classVar = Int("class")
        s = mlp2smt.toSMT(clf, str(classVar), Solver(), "argMaxImpl")

        TestHelper.checkClf(
            self, X, clf, s, classVar, "MLP - HL1 - N2 - argMaxImpl - Digits"
        )
