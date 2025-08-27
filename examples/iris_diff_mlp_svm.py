from z3 import *
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

import mldiff.mlp2smtR as mlp2smt
import mldiff.svm2smtR as svm2smt
from mldiff.diff_matrix import evaluateDiffMatrix

iris = load_iris()
X = iris.data
y = iris.target

# train decision tree
dt = MLPClassifier(max_iter=1000, activation="relu", hidden_layer_sizes=(10))
dt.fit(X, y)

# train svm
svm = LinearSVC()
svm.fit(X, y)

# convert dt and svm to SMT
clDt = Int("classDT")
s = mlp2smt.toSMT(dt, str(clDt), epsilonArgMax=0.001)


clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s, epsilonArgMax=0.001)

# check if the two classifiers disagree
s.push()
s.add(clDt != clSvm)
if s.check() == sat:
    print("The two classifiers disagree on the following model:")
    print(s.model())
else:
    print("The two classifiers agree on all models.")
s.pop()

pp(evaluateDiffMatrix(s, clDt, clSvm, iris.target_names.size))
