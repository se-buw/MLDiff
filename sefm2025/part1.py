from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score 

from z3 import *
import joblib

import mldiff.dt2smtR as dt2smt
import mldiff.svm2smt as svm2smt

# A traditional comparison based on accuracy and F1-scores shows that the DT is preferrable, 
# however, an analysis using MLDiff shows that the DT sometimes misclassifies the poisonous XXX 
# as the XXX used for medical purposes. 
# A corresponding instance is shown below. Note that such an instance is not contained in the 
# dataset, i.e., even an exhaustive search through the dataset would not have revealed the 
# potentially dangerous conflict.

file = open("sefm2025/results/part1_out.txt", "w")

iris = load_iris()
matchingScenarioFound : bool = False

while not matchingScenarioFound:
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)

    # train decision tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    f1_score_dt = f1_score(y_test, dt.predict(X_test), average='macro')    
    accuracy_dt = dt.score(X_test, y_test)

    # train svm
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    f1_score_svm = f1_score(y_test, svm.predict(X_test), average='macro')
    accuracy_svm = svm.score(X_test, y_test)

    # go on, only if the dt is better
    if f1_score_dt > f1_score_svm and accuracy_dt > accuracy_svm:
        # check if poisonous Versicolor is classified as medical Virginica
        classificationPairs = [x for x in zip(dt.predict(iris.data), svm.predict(iris.data) ) if x[0] == 1 and x[1] == 2]
        # go on, 
        if len(classificationPairs) == 0:
            # convert dt and svm to SMT
            clDt = Int("classDT")
            s = dt2smt.toSMT(dt, str(clDt), epsilonDecisionDistance=0.3)
            clSvm = Int("classSVM")
            svm2smt.toSMT(svm, str(clSvm), s)
            # DT should detect virginica and SVM should detect versicolor            
            s.add(clDt == 2, clSvm == 1)

            for i in range(dt.n_features_in_):
                feature = Real("x" + str(i))
                # add min and max
                s.add(feature >= min(iris.data[:,i]), feature <= max(iris.data[:,i]))


            # go on, only if the poisonous Versicolor is classified as medical Virginica
            if s.check() == sat:
                # print f1-scores and accuracy comparison
                file.write('Accuracy DT: ' + str(accuracy_dt)+ '\n' +
                'F1-Score DT: ' + str(f1_score_dt) + '\n' + 
                'Accuracy SVM: ' + str(accuracy_dt) + '\n' + 
                'F1-Score SVM: ' + str(f1_score_svm) + '\n' + 
                'SVM coeff: ' + str(svm.coef_)+ '\n' + 
                'SVM intercept: ' + str(svm.intercept_)+ '\n')
                fig = plt.figure(figsize=(25,20))
                _ = plot_tree(dt, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   label='none')
                fig.savefig("example/paper/part1_dt.pdf", format="pdf")

                # print the matching scenario
                for i in range(dt.n_features_in_):
                    file.write(iris.feature_names[i] + ': ' + str(s.model().eval(Real('x' + str(i)), model_completion=True).as_decimal(prec=3)) + '\n')
                    
                # save the models
                joblib.dump(dt, 'example/paper/part1_dt.joblib')
                joblib.dump(svm, 'example/paper/part1_svm.joblib')
                matchingScenarioFound = True
file.close()