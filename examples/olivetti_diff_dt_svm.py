from z3 import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt


def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))

    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))

    def all_smt_rec(terms):
        if sat == s.check():
            m = s.model()
            yield m
            for i in range(len(terms)):
                s.push()
                block_term(s, m, terms[i])
                for j in range(i):
                    fix_term(s, m, terms[j])
                yield from all_smt_rec(terms[i:])
                s.pop()

    yield from all_smt_rec(list(initial_terms))


faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

# use PCA to reduce the dimensionality
# note that there is a huge dependency between the number of components and the number of classes we can handle
# for all 40 classes, we need ~10 components
# for 10 classes, we can go to ~50 components
preprocessing = PCA(n_components=24)

preprocessing.fit(X)
X = preprocessing.transform(X)

# filter out classes above 10
# not sure whether this should be done before or after PCA
X = X[y <= 10]
y = y[y <= 10]


# train decision tree
dt = DecisionTreeClassifier()
dt.fit(X, y)
print("DT f1-score:", f1_score(y, dt.predict(X), average="macro"))

# train svm
svm = LinearSVC()
svm.fit(X, y)
print("SVM f1-score:", f1_score(y, svm.predict(X), average="macro"))


# convert dt and dt to SMT
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))

clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

features = []
for i in range(X.shape[1]):
    f = Real("x" + str(i))
    features.append(f)
    # s.add( f >= 0, f <= 1)

# check if the two classifiers disagree
s.add(clDt != clSvm)
if s.check() == sat:
    print("The two classifiers disagree on the following model:")
    # for m in all_smt(s, features):
    for m in all_smt(s, [clDt]):
        face = []
        for f in features:
            face.append(float(s.model().eval(f, model_completion=True).as_fraction()))
        # print(s.model())
        face = preprocessing.inverse_transform(face)
        plt.imshow(
            np.array(face).reshape(64, 64),
            interpolation="nearest",
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        plt.ion()
        # plt.show()
        # wait for 100ms
        plt.pause(0.2)
else:
    print("The two classifiers agree on all models.")
