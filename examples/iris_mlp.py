from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

iris = load_iris()
X = iris.data
y = iris.target

clf = MLPClassifier(max_iter=1000, activation="identity")

clf.fit(X, y)

# print(f1_score(y, clf.predict(X), average='macro'))
print(iris.data[0])
# is this the same as for Linear SVM?
print(clf.predict([iris.data[0]]))
# print(toSMT(clf))
