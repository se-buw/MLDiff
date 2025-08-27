import unittest
from sklearn.base import BaseEstimator
from z3 import *

from util import all_smt


class TestHelper:

    @staticmethod
    def old_checkClf(
        test_case, data, classifier, solver, classVar, testedClassifierName
    ):

        print("Execute test for " + testedClassifierName + "\n")

        features: dict = {}
        numFeatures = len(data[0])

        for i in range(numFeatures):
            features.setdefault(i, Real("x" + str(i)))
            print("\nCheck all Data Points:")
        for i in range(len(data)):
            solver.push()
            # set all features
            for f in range(numFeatures):
                solver.add(features[f] == data[i][f])
            # check if the class in SMT is the same as class of the classifier
            prediction = classifier.predict([data[i]])[0]
            val = IntVal(str(prediction))
            solver.add(classVar == val)
            r = solver.check()
            test_case.assertTrue(r == sat)
            solver.pop()

    def checkClf(self, data, classifier, solver, classVar, testedClassifierName):

        print("Execute test for " + testedClassifierName + "\n")

        features: dict = {}
        numFeatures = len(data[0])

        for i in range(numFeatures):
            features.setdefault(i, Real("x" + str(i)))

        print("\nCheck all Data Points:")
        for i in range(len(data)):
            solver.push()
            # set all features
            for f in range(numFeatures):
                solver.add(features[f] == data[i][f])
            # check if the class in SMT is the same as class of the classifier
            prediction = classifier.predict([data[i]])[0]
            r = solver.check()
            self.assertTrue(r == sat)
            preVal = solver.model().eval(classVar).as_long()
            if preVal != prediction:
                prediction = classifier.predict([data[i]])[0]
            self.assertTrue(preVal == prediction)
            solver.pop()

    def checkRoundTrip(
        self, data, classifier: BaseEstimator, solver, classVar, testedClassifierName
    ):

        print("Execute test for " + testedClassifierName + "\n")

        features: dict = {}
        numFeatures = classifier.n_features_in_

        for i in range(numFeatures):
            features.setdefault(i, Real("x" + str(i)))

        print("\nCheck all Data Points:")
        for i in range(len(data)):
            solver.push()
            # set all features
            for f in range(numFeatures):
                solver.add(features[f] == data[i][f])
            # check if the class in SMT is the same as class of the classifier
            predictionOrig = classifier.predict([data[i]])[0]
            r = solver.check()
            self.assertTrue(r == sat)
            preVal = solver.model().eval(classVar).as_long()
            self.assertTrue(preVal == predictionOrig)

            dataExtracted = [
                float(
                    solver.model()
                    .eval(features[f], model_completion=True)
                    .as_fraction()
                )
                for f in range(numFeatures)
            ]
            predictionExtracted = classifier.predict([dataExtracted])[0]
            self.assertTrue(predictionExtracted == predictionOrig)
            solver.pop()

    def checkAllClasses(
        self, classifier: BaseEstimator, solver, classVar, testedClassifierName
    ):

        print("Execute test for " + testedClassifierName + "\n")

        features: dict = {}
        numFeatures = classifier.n_features_in_

        for i in range(numFeatures):
            features.setdefault(i, Real("x" + str(i)))

        for m in all_smt(solver, [classVar]):
            preVal = m.eval(classVar).as_long()
            dataExtracted = [
                float(m.eval(features[f], model_completion=True).as_fraction())
                for f in range(numFeatures)
            ]
            predictionExtracted = classifier.predict([dataExtracted])[0]
            print(
                f"Current model preVal: {preVal} predictionExtracted: {predictionExtracted}"
            )
            self.assertTrue(predictionExtracted == preVal)

    def checkAllClassCombinations(
        self, clf1: BaseEstimator, clf2: BaseEstimator, s: Solver, clVar1, clVar2
    ):

        features: dict = {}
        numFeatures = clf1.n_features_in_

        for i in range(numFeatures):
            features.setdefault(i, Real("x" + str(i)))
        if s.check() == sat:
            for m in all_smt(s, [clVar1, clVar2]):
                preVal1 = m.eval(clVar1).as_long()
                preVal2 = m.eval(clVar2).as_long()
                dataExtracted = [
                    float(m.eval(features[f], model_completion=True).as_fraction())
                    for f in range(numFeatures)
                ]
                predictionExtracted1 = clf1.predict([dataExtracted])[0]
                predictionExtracted2 = clf2.predict([dataExtracted])[0]
                print(
                    f"Current model preVal1: {preVal1}, preVal2: {preVal2} predictionExtracted1: {predictionExtracted1}, predictionExtracted2: {predictionExtracted2}"
                )
                self.assertTrue(predictionExtracted1 == preVal1)
                self.assertTrue(predictionExtracted2 == preVal2)

    def checkCombinedClf(
        self,
        data,
        classifier1,
        classifier2,
        classVar1,
        classVar2,
        solver,
        classifierFileNames,
    ):

        print("Execute test for " + classifierFileNames[0] + "against")
        print(classifierFileNames[1] + " and " + classifierFileNames[2] + "\n")

        features: dict = {}
        numFeatures = len(data[0])

        for i in range(numFeatures):
            features.setdefault(i, Real("x" + str(i)))
        print("\nCheck all Data Points:")
        for i in range(len(data)):
            solver.push()
            # set all features
            for f in range(numFeatures):
                solver.add(features[f] == data[i][f])

            prediction1 = classifier1.predict([data[i]])[0]
            prediction2 = classifier2.predict([data[i]])[0]
            r = solver.check()
            self.assertTrue(r == sat)

            preVal1 = solver.model().eval(classVar1).as_long()
            preVal2 = solver.model().eval(classVar2).as_long()

            if preVal1 != prediction1:
                classifier1.predict([data[i]])[0]

            self.assertEqual(preVal1, prediction1)
            self.assertEqual(preVal2, prediction2)
            solver.pop()
