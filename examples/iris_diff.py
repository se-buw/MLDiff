from z3 import *
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
from mldiff.diff_matrix import evaluateDiffMatrix

iris = load_iris()
X = iris.data
y = iris.target

# train decision tree
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X, y)

# train svm
svm = LinearSVC()
svm.fit(X, y)

# convert dt and svm to SMT
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))


clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

# check if the two classifiers disagree
s.push()
s.add(clDt != clSvm)
if s.check() == sat:
    print("The two classifiers disagree on the following model:")
    print(s.model())
else:
    print("The two classifiers agree on all models.")
s.pop()

pp(evaluateDiffMatrix(s, clDt, clSvm, iris.target_names.size))

pp(evaluateDiffMatrix(s, clSvm, clSvm, iris.target_names.size))

pp(evaluateDiffMatrix(s, clDt, clDt, iris.target_names.size))
