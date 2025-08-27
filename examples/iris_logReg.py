from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X = iris.data
y = iris.target

clf = LogisticRegression()

clf.fit(X, y)

# print(f1_score(y, clf.predict(X), average='macro'))
print(iris.data[0])
# is this the same as for Linear SVM?
print(clf.predict([iris.data[0]]))
# print(toSMT(clf))
