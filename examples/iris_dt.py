from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mldiff.dt2smt import toSMT

iris = load_iris()
X = iris.data
y = iris.target
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# print(f1_score(y, clf.predict(X), average='macro'))
print(iris.data[0])
print(clf.predict([iris.data[0]]))
print(toSMT(clf))
