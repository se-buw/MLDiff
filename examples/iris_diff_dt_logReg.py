from z3 import *
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import mldiff.dt2smt as dt2smt
import mldiff.logReg2smt as logReg2smt
from mldiff.diff_matrix import evaluateDiffMatrix

iris = load_iris()
X = iris.data
y = iris.target

# train decision tree
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X, y)

# train logReg
logReg = LogisticRegression()
logReg.fit(X, y)

# convert dt and svm to SMT
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))

clLogReg = Int("classLogReg")
logReg2smt.toSMT(logReg, str(clLogReg), s)

# check if the two classifiers disagree
s.push()
s.add(clDt != clLogReg)
if s.check() == sat:
    print("The two classifiers disagree on the following model:")
    print(s.model())
else:
    print("The two classifiers agree on all models.")
s.pop()

pp(evaluateDiffMatrix(s, clDt, clLogReg, iris.target_names.size))
