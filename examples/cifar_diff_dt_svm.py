from z3 import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, mean_squared_error

import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
from examples.cifar import load_cifar10


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


faces = load_cifar10(data_dir="data/cifar-10-batches-py")
X, y = faces.data, faces.target
preprocessing = PCA(n_components=434)
preprocessing.fit(X)

# check quality of PCA
print("PCA explained variance ratio:", sum(preprocessing.explained_variance_ratio_))


X = preprocessing.transform(X)
Xinv = preprocessing.inverse_transform(X)
reconstruction_error = mean_squared_error(faces.data, Xinv)
print("Reconstruction error:", reconstruction_error)
print("Number of components used:", preprocessing.n_components_)

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

# check if the two classifiers disagree
s.add(clDt != clSvm)
if s.check() == sat:
    print("The two classifiers disagree on the following model:")
    for m in all_smt(s, [clDt]):
        face = []
        for f in features:
            face.append(float(s.model().eval(f, model_completion=True).as_fraction()))
        # print(s.model())
        face = preprocessing.inverse_transform(face)
        plt.imshow(
            face.reshape(3, 32, 32).astype(np.uint8).transpose(1, 2, 0),
            interpolation="nearest",
            vmin=0,
            vmax=255,
        )
        plt.ion()
        plt.pause(0.2)
else:
    print("The two classifiers agree on all models.")
