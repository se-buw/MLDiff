import unittest
from z3 import *
from sklearn.calibration import LinearSVC
from sklearn.conftest import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from mldiff import svm2smt
import mldiff.logReg2smt as logReg2smt

from tests.helper import TestHelper


class TestClf2SmtIris(unittest.TestCase):

    @unittest.skip
    def test_iris_dt(self):
        faces = fetch_olivetti_faces()
        X, y = faces.data, faces.target

        preprocessing = PCA(n_components=24)

        preprocessing.fit(X)
        X = preprocessing.transform(X)

        clf1 = LogisticRegression()
        clf1.fit(X, y)

        clf2 = LinearSVC()
        clf2.fit(X, y)

        classVar1 = Int("class1")
        s = logReg2smt.toSMT(clf1, str(classVar1), Solver())
        classVar2 = Int("class2")
        s = svm2smt.toSMT(clf2, str(classVar2), s)

        TestHelper.checkClf(self, X, clf1, s, classVar1, "DT - Olivetti")
        TestHelper.checkClf(self, X, clf2, s, classVar2, "DT - Olivetti")

    @unittest.skip
    def test_olivetti_smt_round(self):
        faces = fetch_olivetti_faces()
        X, y = faces.data, faces.target

        preprocessing = PCA(n_components=24)

        preprocessing.fit(X)
        X = preprocessing.transform(X)

        clf1 = LogisticRegression()
        clf1.fit(X, y)

        clf2 = LinearSVC()
        clf2.fit(X, y)

        classVar1 = Int("class1")
        s = logReg2smt.toSMT(clf1, str(classVar1), Solver())
        classVar2 = Int("class2")
        s = svm2smt.toSMT(clf2, str(classVar2), s)

        TestHelper.checkRoundTrip(self, X, clf1, s, classVar1, "DT - Olivetti")
        TestHelper.checkRoundTrip(self, X, clf2, s, classVar2, "DT - Olivetti")

    # @unittest.skip
    def test_olivetty_exhaust(self):
        faces = fetch_olivetti_faces()
        X, y = faces.data, faces.target

        preprocessing = PCA(n_components=24)

        preprocessing.fit(X)
        X = preprocessing.transform(X)

        X = X[y < 10]
        y = y[y < 10]

        clf1 = LogisticRegression()
        clf1.fit(X, y)

        clf2 = LinearSVC()
        clf2.fit(X, y)

        classVar1 = Int("class1")
        s = logReg2smt.toSMT(clf1, str(classVar1), Solver())
        classVar2 = Int("class2")
        s = svm2smt.toSMT(clf2, str(classVar2), s)

        TestHelper.checkAllClasses(self, clf1, s, classVar1, "DT - Olivetti")
        TestHelper.checkAllClasses(self, clf2, s, classVar2, "DT - Olivetti")

    @unittest.skip
    def test_olivetty_exhaustCombination(self):
        faces = fetch_olivetti_faces()
        X, y = faces.data, faces.target

        preprocessing = PCA(n_components=24)

        preprocessing.fit(X)
        X = preprocessing.transform(X)

        X = X[y < 10]
        y = y[y < 10]

        clf1 = LogisticRegression()
        clf1.fit(X, y)

        clf2 = LinearSVC()
        clf2.fit(X, y)

        classVar1 = Int("class1")
        s = logReg2smt.toSMT(clf1, str(classVar1), Solver())
        classVar2 = Int("class2")
        s = svm2smt.toSMT(clf2, str(classVar2), s)

        TestHelper.checkAllClassCombinations(self, clf1, clf2, s, classVar1, classVar2)


if __name__ == "__main__":
    unittest.main()
