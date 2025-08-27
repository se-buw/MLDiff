from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from z3 import *
from mldiff.svm2smt import toSMT

iris = load_digits()
X = iris.data
y = iris.target
clf = LinearSVC()
clf.fit(X, y)
s: Solver = toSMT(clf)
print(s.sexpr())
