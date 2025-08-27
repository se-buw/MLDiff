from z3 import *
import joblib
from matplotlib import pyplot as plt
import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris

iris_dt = joblib.load('sefm2025/part1_dt.joblib')
iris_svm = joblib.load('sefm2025/part1_svm.joblib')


dt = dt2smt.toSMT(iris_dt, "classDT").sexpr()
svm = svm2smt.toSMT(iris_svm, "classSVM").sexpr()

with open("sefm2025/results/dt.smt2", "w") as f:
    f.write(dt)

with open("sefm2025/results/svm.smt2", "w") as f:
    f.write(svm)

# get features from the iris dataset
iris = load_iris()
feature_names = iris.feature_names
class_names = iris.target_names
# plot tree
fig = plt.figure(figsize=(6,8))
_ = plot_tree(iris_dt, 
    feature_names=feature_names,  
    class_names=class_names,
    label='none')
plt.show()