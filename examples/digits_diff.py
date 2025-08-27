from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from decimal import Decimal

from z3 import *

import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt

from mldiff.diff_matrix import *


digits = load_digits()
X = digits.data
y = digits.target

# train decision tree
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X, y)
print("DT f1-score:", f1_score(y, dt.predict(X), average="macro"))

# train svm
svm = LinearSVC()
svm.fit(X, y)

print("SVM f1-score:", f1_score(y, svm.predict(X), average="macro"))


# convert dt and svm to SMT
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))

clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

for i in range(digits.data.shape[1]):
    f = Real("x" + str(i))


def confirm(s: Solver, class1: ArithRef, clf1, class2: ArithRef, clf2):
    dataFromModel = [[]]
    for i in range(digits.data.shape[1]):
        f = Real("x" + str(i))
        frac = s.model().eval(f, model_completion=True).as_fraction()
        fVal = round(float(frac.numerator) / float(frac.denominator))
        if fVal < 0:
            fVal = 0
        if fVal > 16:
            fVal = 16
        dataFromModel[0].append(fVal)

    confirm1 = clf1.predict(dataFromModel)[0] == s.model().eval(class1).as_long()
    confirm2 = clf2.predict(dataFromModel)[0] == s.model().eval(class2).as_long()
    return confirm1 and confirm2


def identify(s: Solver, clf1, clf2):
    dataFromModel = [[]]
    for i in range(digits.data.shape[1]):
        f = Real("x" + str(i))
        frac = s.model().eval(f, model_completion=True).as_fraction()
        fVal = round(Decimal(frac.numerator) / Decimal(frac.denominator))
        if fVal < 0:
            fVal = 0
        if fVal > 16:
            fVal = 16
        dataFromModel[0].append(fVal)
    return clf1.predict(dataFromModel)[0], clf2.predict(dataFromModel)[0]


def block(s: Solver):
    intervalsToBlock = []
    for i in range(digits.data.shape[1]):
        f = Real("x" + str(i))
        frac = s.model().eval(f, model_completion=True).as_fraction()
        fVal = round(Decimal(frac.numerator) / Decimal(frac.denominator))
        if fVal <= 0:
            intervalsToBlock.append(f < 0.5)
        elif fVal >= 16:
            intervalsToBlock.append(f >= 15.5)
        else:
            intervalsToBlock.append(And(f >= fVal - 0.5, f < fVal + 0.5))
    # print(intervalsToBlock)
    s.add(Not(And(intervalsToBlock)))


print(
    evaluateDiffMatrixIdentify(
        s, clDt, dt, clSvm, svm, digits.target_names.size, identify, block
    )
)
