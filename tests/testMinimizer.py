import time
import unittest
from sklearn.datasets import load_iris, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
import mldiff.logReg2smt as logReg2smt
import mldiff.mlp2smt as mlp2smt
import mldiff.minimizer as m
from tests.util import all_smt

from z3 import *


class TestMinimizer(unittest.TestCase):

    # @unittest.skip
    def test_iris_dt(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        # clf = LinearSVC()
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        classVar = Int("class")
        s = dt2smt.toSMT(clf, str(classVar), Solver())

        # Create the goal
        for i in range(len(clf.classes_)):
            goal = classVar == i
            # extract the elements
            elements = self.extractElements(goal, s)
            lenElem = len(elements)
            if lenElem != 0:
                # minimize the elements
                min = m.minimize(s, elements, goal)
                lenMin = len(min)
                # check if the minimization was successful
                print("lenElem: ", lenElem, " lenMin: ", lenMin)
                self.assertTrue(lenMin <= lenElem)

    def test_iris_dt_dt(self):
        self.fitCompareAndMin(
            load_iris(), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier()
        )

    def test_digits_dt_dt(self):
        self.fitCompareAndMin(
            load_digits(), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier()
        )

    def test_iris_dt_svm(self):
        self.fitCompareAndMin(
            load_iris(), DecisionTreeClassifier(max_depth=3), LinearSVC()
        )

    def test_digits_dt_svm(self):
        self.fitCompareAndMin(
            load_digits(), DecisionTreeClassifier(max_depth=3), LinearSVC()
        )

    def test_iris_dt_logReg(self):
        self.fitCompareAndMin(
            load_iris(), DecisionTreeClassifier(), LogisticRegression(max_iter=1000)
        )

    def test_digits_dt_logReg(self):
        self.fitCompareAndMin(
            load_digits(), DecisionTreeClassifier(), LogisticRegression(max_iter=1000)
        )

    def test_iris_svm_logReg(self):
        self.fitCompareAndMin(
            load_iris(), LinearSVC(), LogisticRegression(max_iter=1000)
        )

    def test_digits_svm_logReg(self):
        self.fitCompareAndMin(
            load_digits(), LinearSVC(), LogisticRegression(max_iter=1000)
        )

    def blockClasses(
        self, s: Solver, elements: list[ExprRef], classVar1: ExprRef, classVar2: ExprRef
    ):
        comboBlock = []
        s.push()
        s.add(And(elements))
        if s.check() == sat:
            for m in all_smt(s, [classVar1, classVar2]):
                c1Val = classVar1 == m.eval(classVar1)
                c2Val = classVar2 == m.eval(classVar2)
                comboBlock.append(Not(And(c1Val, c2Val)))
        s.pop()
        # block the current classes from the model
        s.add(comboBlock)

    def extractElements(self, goal, s) -> list:
        s.push()

        elements: list = []

        # find a model that satisfies the goal
        s.add(goal)
        if s.check() == sat:
            # extract the elements from the current model satisfying the goal
            decls = s.model().decls()
            for i in range(len(decls)):
                # only use feature variables (starting with x)
                if decls[i].name().startswith("x"):
                    f = Real(decls[i].name())
                    val = s.model().eval(f)
                    elements.append(f == val)

        s.pop()
        return elements

    def fitCompareAndMin(self, data, clf1, clf2):
        X = data.data
        y = data.target

        clf1.fit(X, y)
        clf2.fit(X, y)
        classVar1 = Int("class1")
        classVar2 = Int("class2")
        s = self.toSmt(clf1, classVar1, Solver())
        s = self.toSmt(clf2, classVar2, s)
        s.set("threads", 8)

        done: bool = False
        while not done:
            goal = classVar1 != classVar2
            # extract the elements
            elements = self.extractElements(goal, s)
            lenElem = len(elements)
            if lenElem != 0:
                # minimize the elements
                # compare running time of minimizeCore and minimize
                start = time.time_ns()
                min1 = m.minimizeCore(s, elements, goal)
                tCore = time.time_ns() - start
                start = time.time_ns()
                min2 = m.minimize(s, elements, goal)
                tDDMin = time.time_ns() - start
                # check if the minimization was successful
                if tDDMin == 0:
                    ratio = "inf"
                else:
                    ratio = round(tCore / tDDMin, 5)
                print(
                    "lenElem: ",
                    lenElem,
                    " lenMinCore: ",
                    len(min1),  # " tCore: ", tCore,
                    " lenMinDDMin: ",
                    len(min2),  # " tDDMin: ", tDDMin,
                    " tCore/tDDMin: ",
                    ratio,
                )
                self.assertTrue(len(min1) <= lenElem and len(min2) <= lenElem)
                self.blockClasses(s, min1, classVar1, classVar2)
            else:
                done = True

    def toSmt(self, clf, classVar, s):
        if isinstance(clf, DecisionTreeClassifier):
            return dt2smt.toSMT(clf, str(classVar), s)
        elif isinstance(clf, LinearSVC):
            return svm2smt.toSMT(clf, str(classVar), s)
        elif isinstance(clf, LogisticRegression):
            return logReg2smt.toSMT(clf, str(classVar), s)
        elif isinstance(clf, MLPClassifier):
            return mlp2smt.toSMT(clf, str(classVar), s)
        else:
            raise Exception("Unknown classifier type")
