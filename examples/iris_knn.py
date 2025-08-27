from sklearn.datasets import load_iris
from sklearn.neighbors import KDTree, KNeighborsClassifier

# from mldiff.dt2smt import toSMT

iris = load_iris()
X = iris.data
y = iris.target

clf = KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree")

clf.fit(X, y)

t: KDTree = clf._tree
# traverse the KDTree of the classifier


print(t.get_tree_stats())
print(t.query([iris.data[0]], k=3))


# print(f1_score(y, clf.predict(X), average='macro'))
print(iris.data[0])
print(clf.predict([iris.data[0]]))
# print(toSMT(clf))
