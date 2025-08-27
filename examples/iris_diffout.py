from z3 import *
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
import mldiff.if2smt as if2smt
from mldiff.util import all_smt


iris = load_iris()
X = iris.data
y = iris.target

# train decision tree
dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(X, y)

# train svm
svm = LinearSVC(random_state=42)
# fit with reverse class labels
ySwap = np.where(y == 0, 1, 0)
# fit with swapped 1 - 0 class labels
svm.fit(X, ySwap)

# convert dt and svm to SMT
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))


clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

# outlier detection for dataset
od = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
od.fit(X)

# add outlier suppression to the SMT solver
if2smt.toSMT(od, s)

features = []
for i in range(iris.data.shape[1]):
    f = Real("x" + str(i))
    features.append(f)

# check if the two classifiers disagree
s.push()
s.add(clDt != clSvm)

# iterate over models found and check if they are outliers

if s.check() == sat:
    print("The two classifiers disagree.")
    for m in all_smt(s, [clDt]):
        data = []
        for f in features:
            data.append(float(s.model().eval(f, model_completion=True).as_fraction()))
        # check if the model is an outlier
        data = np.array(data).reshape(1, -1)
        outlier = od.predict(data)[0] == -1
        if outlier:
            print("The model is an outlier:")
        else:
            print("The model is plausible:")
        print(data)
else:
    print("The two classifiers agree on all models.")
