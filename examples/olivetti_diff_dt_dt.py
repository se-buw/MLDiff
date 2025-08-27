from z3 import *
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

import mldiff.dt2smt as dt2smt


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

# train decision tree
dt = DecisionTreeClassifier(max_depth=30)
dt.fit(X, y)
print("DT f1-score:", f1_score(y, dt.predict(X), average="macro"))

dt2 = DecisionTreeClassifier(max_depth=28)
dt2.fit(X, y)
print("DT2 f1-score:", f1_score(y, dt2.predict(X), average="macro"))

# convert dt and dt to SMT
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))

clDt2 = Int("classSVM")
dt2smt.toSMT(dt2, str(clDt2), s)

features = []
for i in range(faces.data.shape[1]):
    f = Real("x" + str(i))
    features.append(f)
    s.add(f >= 0, f <= 1)

# check if the two classifiers disagree
s.add(clDt != clDt2)
if s.check() == sat:
    print("The two classifiers disagree on the following model:")
    # for m in all_smt(s, features):
    for m in all_smt(s, [clDt]):
        face = []
        for f in features:
            face.append(float(s.model().eval(f, model_completion=True).as_fraction()))
        # print(s.model())
        from matplotlib import pyplot as plt

        plt.imshow(
            np.array(face).reshape(64, 64),
            interpolation="nearest",
            cmap="gray_r",
            vmin=0,
            vmax=1,
        )
        plt.ion()
        # wait for 100ms
        plt.pause(0.2)
else:
    print("The two classifiers agree on all models.")
