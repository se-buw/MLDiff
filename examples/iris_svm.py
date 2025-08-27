from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

from mldiff.svm2smt import toSMT

iris = load_iris()
X = iris.data
y = iris.target
clf = LinearSVC()
clf.fit(X, y)

print(toSMT(clf).sexpr())
