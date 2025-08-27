from z3 import *
import numpy as np
from sklearn.datasets import load_digits
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


digits = load_digits()
X = digits.data
y = digits.target

# train decision tree
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X, y)
print("DT f1-score:", f1_score(y, dt.predict(X), average="macro"))

# train svm
svm = LinearSVC()
svm.fit(X, y)
print("SVM f1-score:", f1_score(y, svm.predict(X), average="macro"))

# convert dt and svm to SMT
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))

clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

features = []
for i in range(digits.data.shape[1]):
    f = Real("x" + str(i))
    features.append(f)
    s.add(IsInt(f))
    s.add(f >= 0, f <= 16)

# check if the two classifiers disagree
s.add(clDt != clSvm)
if s.check() == sat:
    print("The two classifiers disagree on the following model:")
    # for m in all_smt(s, features):
    for m in all_smt(s, [clDt]):
        data = []
        for f in features:
            data.append(float(s.model().eval(f, model_completion=True).as_fraction()))
        # print(s.model())
        from matplotlib import pyplot as plt

        plt.imshow(
            np.array(data).reshape(8, 8),
            interpolation="nearest",
            cmap="gray_r",
            vmin=0,
            vmax=16,
        )
        plt.ion()
        # wait for 100ms
        plt.pause(0.1)
else:
    print("The two classifiers agree on all models.")
