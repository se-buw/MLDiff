from z3 import *
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
from mldiff.diff_matrix import evaluateDiffMatrix

# Load data and train classifiers
iris = load_iris()
X, y = iris.data, iris.target

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X, y)

svm = LinearSVC()
svm.fit(X, y)

# Convert to SMT formulas
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))

clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

# Check for disagreements
s.push()
s.add(clDt != clSvm)
if s.check() == sat:
    print("Classifiers disagree on:", s.model())
else:
    print("Classifiers agree on all inputs")
s.pop()