import time
import sys

sys.set_int_max_str_digits(0)

import argparse

import joblib
from sklearn.decomposition import PCA
from sklearn.datasets import (
    fetch_olivetti_faces,
    load_iris,
    load_digits,
    load_breast_cancer,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, mean_squared_error

from matplotlib import pyplot as plt
import numpy as np

from z3 import *

import mldiff.dt2smtR as dt2smt
import mldiff.svm2smtR as svm2smt
import mldiff.logReg2smtR as logReg2smt
import mldiff.mlp2smtR as mlp2smt


# SETTINGS
# ------------------------------------
VIZ_FLAG = False  # Visualization flag
PCA_FLAG = False  # PCA flag. Apply PCA to the dataset if True.
FEATURE_CON_FLAG = False  # Domain constraints flag
DATASET = ""
CLASSIFIER1 = ""
CLASSIFIER2 = ""
EPSILON_DECISION_DISTANCE = 0.0  # Used for Decision Tree
EPSILON_ARGMAX = 0.0  # Used for SVM, Logistic Regression, MLP
SAVE_SUB_DIR = ""
HIDDEN_LAYERS = 10
# ------------------------------------


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


def load_dataset(dataset):
    """Load the dataset based on the input string.

    params:
    dataset: str - dataset type. Options: 'iris', 'digits', 'olivetti', 'cancer'

    returns:
    dataset: dataset object in sklearn format (data, target)
    """
    if dataset == "iris":
        return load_iris()
    elif dataset == "digits":
        return load_digits()
    elif dataset == "olivetti":
        return fetch_olivetti_faces()
    elif dataset == "cancer":
        return load_breast_cancer()
    else:
        raise ValueError("Invalid dataset. Options: iris, digits, olivetti, cancer")


def load_classifier(classifier):
    """Load the classifier based on the input string.

    params:
    classifier: str - classifier type. Options: 'dt', 'svm', 'logreg', 'mlp'

    returns:
    clf: classifier object
    """
    if classifier == "dt":
        return DecisionTreeClassifier()
    elif classifier == "svm":
        return LinearSVC()
    elif classifier == "logreg":
        return LogisticRegression()
    elif classifier == "mlp":
        return MLPClassifier(max_iter=100, hidden_layer_sizes=HIDDEN_LAYERS)
    else:
        raise ValueError("Invalid classifier. Options: dt, svm, logreg, mlp")


def visualize_disagreements(
    clfPredictions_x,
    clfPredictions_y,
    misclassifiedSolverDisagreements_x,
    misclassifiedSolverDisagreements_y,
    correctSolverDisagreements_x,
    correctSolverDisagreements_y,
):
    """Visualize the disagreements between the two classifiers."""

    # Plot the data
    plt.plot(
        clfPredictions_x,
        clfPredictions_y,
        "s",
        markerfacecolor="none",
        label="Classifier Predictions",
    )
    plt.plot(
        misclassifiedSolverDisagreements_x,
        misclassifiedSolverDisagreements_y,
        "o",
        markerfacecolor="none",
        label="Misclassified Solver Disagreements",
    )
    plt.plot(
        correctSolverDisagreements_x,
        correctSolverDisagreements_y,
        "x",
        markerfacecolor="none",
        label="Correct Solver Disagreements",
    )

    # Add labels and legend
    plt.xlabel("Classifier 1")
    plt.ylabel("Classifier 2")
    plt.title("Disagreements between the two classifiers")
    plt.legend(loc="best")

    # Show the plot
    plt.show()


def get_model(clf, X_train, y_train):
    """Get trained model from models directory.
    If it does not exist, train it and save it to models directory.
    Modify the global settings to change the model directory and the model name.

    params:
    clf: str - classifier type. Options: 'dt', 'svm', 'logreg', 'mlp'
    `X_train`: numpy array - training data
    `y_train`: numpy array - training labels

    returns:
    `model`: fitted model
    """
    model_dir = f"models/"
    os.makedirs(model_dir, exist_ok=True)
    m = f"_{HIDDEN_LAYERS}" if clf == "mlp" else ""
    model_path = f"{model_dir}{DATASET}_{clf}{m}{'_withPCA' if PCA_FLAG else ''}.joblib"

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model
    else:
        print(f"Model not found. Training model...")
        model = load_classifier(clf)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        return model


def addFeatureConstraints(s: Solver, features, dataset, pca: PCA = None):
    if dataset == "olivetti":
        minPca = pca.transform(np.zeros((1, 4096)))[0]
        print(minPca)
        maxPca = pca.transform(np.ones((1, 4096)))[0]
        print(maxPca)
        for i in range(len(features)):
            lower = min(minPca[i], maxPca[i])
            upper = max(minPca[i], maxPca[i])
            s.add(features[i] >= lower, features[i] <= upper)
    elif dataset == "iris":
        for i in range(len(features)):
            s.add(features[i] > 0)
    elif dataset == "digits":
        for i in range(len(features)):
            s.add(features[i] >= 0, features[i] <= 16, IsInt(features[i]))
    elif dataset == "cancer":
        for i in range(len(features)):
            s.add(features[i] > 0)
    else:
        raise ValueError("Invalid dataset. Options: iris, digits, olivetti, cancer")


def main():
    global DATASET, CLASSIFIER1, CLASSIFIER2, PCA_FLAG, FEATURE_CON_FLAG, EPSILON_DECISION_DISTANCE, EPSILON_ARGMAX, SAVE_SUB_DIR, HIDDEN_LAYERS, VIZ_FLAG

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=DATASET,
        help="Dataset type. Options: 'iris', 'digits', 'olivetti', 'cancer'",
    )
    arg_parser.add_argument(
        "-c1",
        "--classifier1",
        type=str,
        default=CLASSIFIER1,
        help="Classifier type. Options: 'dt', 'svm', 'logreg', 'mlp'",
    )
    arg_parser.add_argument(
        "-c2",
        "--classifier2",
        type=str,
        default=CLASSIFIER2,
        help="Classifier type. Options: 'dt', 'svm', 'logreg', 'mlp'",
    )
    arg_parser.add_argument(
        "-pca",
        "--pca",
        action="store_true",
        default=PCA_FLAG,
        help="Apply PCA to the dataset",
    )
    arg_parser.add_argument(
        "-fc",
        "--feature-constraint",
        action="store_true",
        help="Apply domain constraints to the features",
    )
    arg_parser.add_argument(
        "-ed",
        "--epsilon-decision-distance",
        type=float,
        default=EPSILON_DECISION_DISTANCE,
        help="Epsilon decision distance for Decision Tree",
    )
    arg_parser.add_argument(
        "-ea",
        "--epsilon-argmax",
        type=float,
        default=EPSILON_ARGMAX,
        help="Epsilon argmax for SVM, Logistic Regression, MLP",
    )
    arg_parser.add_argument(
        "-sd",
        "--save-dir",
        type=str,
        default=SAVE_SUB_DIR,
        help="Save directory. Example: 'RQ2'",
    )
    arg_parser.add_argument(
        "-v",
        "--visualization",
        action="store_true",
        help="Enable visualization of disagreements",
    )
    args = arg_parser.parse_args()

    # Update settings
    DATASET = args.dataset
    CLASSIFIER1 = args.classifier1
    CLASSIFIER2 = args.classifier2
    PCA_FLAG = args.pca
    FEATURE_CON_FLAG = args.feature_constraint
    EPSILON_DECISION_DISTANCE = args.epsilon_decision_distance
    EPSILON_ARGMAX = args.epsilon_argmax
    SAVE_SUB_DIR = args.save_dir
    VIZ_FLAG = args.visualization

    if DATASET == "olivetti" or DATASET == "iris":
        HIDDEN_LAYERS = 20

    # Load dataset
    dataset = load_dataset(DATASET)
    X, y = dataset.data, dataset.target

    res_dir = f"results/{SAVE_SUB_DIR}/"
    os.makedirs(res_dir, exist_ok=True)
    m = f"_{HIDDEN_LAYERS}" if CLASSIFIER1 == "mlp" or CLASSIFIER2 == "mlp" else ""
    res = open(f"{res_dir}{DATASET}_{CLASSIFIER1}_{CLASSIFIER2}{m}_results.txt", "w")

    res.write(
        f"Settings:\nDataset: {DATASET}\nClassifier1: {CLASSIFIER1}\nClassifier2: {CLASSIFIER2}\nPCA: {PCA_FLAG}\nDomain constraints: {FEATURE_CON_FLAG}\nEpsilon decision distance: {EPSILON_DECISION_DISTANCE}\nEpsilon argmax: {EPSILON_ARGMAX}\n\n"
    )

    # Apply PCA only if PCA_FLAG is enabled
    if PCA_FLAG:
        if os.path.exists(f"models/{DATASET}_pca.joblib"):
            print(f"Loading PCA from models/{DATASET}_pca.joblib")
            pca = joblib.load(f"models/{DATASET}_pca.joblib")
        else:
            print("Training PCA...")
            pca = PCA(n_components=24)
            pca.fit(X)
            joblib.dump(pca, f"models/{DATASET}_pca.joblib")
        print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_)}")
        res.write(
            f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_)}\n"
        )
        X = pca.transform(X)
        Xinv = pca.inverse_transform(X)
        reconstruction_error = mean_squared_error(dataset.data, Xinv)
        print(f"Reconstruction error: {reconstruction_error}")
        res.write(f"Reconstruction error: {reconstruction_error}\n")

    # Take only the first 10 classes for the olivetti dataset
    if DATASET == "olivetti":
        X = X[y < 10]
        y = y[y < 10]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load classifier
    clf1 = get_model(CLASSIFIER1, X_train, y_train)
    clf2 = get_model(CLASSIFIER2, X_train, y_train)

    print(f"Classifier {CLASSIFIER1} accuracy: {clf1.score(X_test, y_test)}")
    res.write(f"Classifier {CLASSIFIER1} accuracy: {clf1.score(X_test, y_test)}\n")
    print(
        f"Classifier {CLASSIFIER1} f1-score: {f1_score(y, clf1.predict(X), average='macro')}"
    )
    res.write(
        f"Classifier {CLASSIFIER1} f1-score: {f1_score(y, clf1.predict(X), average='macro')}\n"
    )
    print(f"Classifier {CLASSIFIER2} accuracy: {clf2.score(X_test, y_test)}")
    res.write(f"Classifier {CLASSIFIER2} accuracy: {clf2.score(X_test, y_test)}\n")
    print(
        f"Classifier {CLASSIFIER2} f1-score: {f1_score(y, clf2.predict(X), average='macro')}"
    )
    res.write(
        f"Classifier {CLASSIFIER2} f1-score: {f1_score(y, clf2.predict(X), average='macro')}\n\n"
    )

    # Translate classifiers to SMT
    cl1 = Int("class1")
    s = Solver()
    if CLASSIFIER1 == "dt":
        s = dt2smt.toSMT(
            clf1, str(cl1), s, epsilonDecisionDistance=EPSILON_DECISION_DISTANCE
        )
    elif CLASSIFIER1 == "svm":
        s = svm2smt.toSMT(clf1, str(cl1), s, epsilonArgMax=EPSILON_ARGMAX)
    elif CLASSIFIER1 == "logreg":
        s = logReg2smt.toSMT(clf1, str(cl1), s, epsilonArgMax=EPSILON_ARGMAX)
    elif CLASSIFIER1 == "mlp":
        s = mlp2smt.toSMT(clf1, str(cl1), s, epsilonArgMax=EPSILON_ARGMAX)
    else:
        raise ValueError("Invalid classifier. Options: dt, svm, logreg, mlp")

    cl2 = Int("class2")
    if CLASSIFIER2 == "dt":
        dt2smt.toSMT(
            clf2, str(cl2), s, epsilonDecisionDistance=EPSILON_DECISION_DISTANCE
        )
    elif CLASSIFIER2 == "svm":
        svm2smt.toSMT(clf2, str(cl2), s, epsilonArgMax=EPSILON_ARGMAX)
    elif CLASSIFIER2 == "logreg":
        logReg2smt.toSMT(clf2, str(cl2), s, epsilonArgMax=EPSILON_ARGMAX)
    elif CLASSIFIER2 == "mlp":
        mlp2smt.toSMT(clf2, str(cl2), s, epsilonArgMax=EPSILON_ARGMAX)
    else:
        raise ValueError("Invalid classifier. Options: dt, svm, logreg, mlp")

    features = []
    for i in range(X.shape[1]):
        f = Real("x" + str(i))
        features.append(f)
    if FEATURE_CON_FLAG:
        if PCA_FLAG:
            addFeatureConstraints(s, features, DATASET, pca)
        else:
            addFeatureConstraints(s, features, DATASET)

    # Check if the two classifiers disagree
    s.add(cl1 != cl2)
    start_time = time.time()
    if s.check() == sat:
        print("The two classifiers disagree on the following model:")
        solverDisagreements = []
        misclassifiedSolverDisagreements = []
        correctSolverDisagreements = []
        spuriousSolverDisagreements = []
        for m in all_smt(s, [cl1, cl2]):
            data = []
            for f in features:
                data.append(float(m.eval(f, model_completion=True).as_fraction()))
            cl1_pred_smt = m.eval(cl1, model_completion=True).as_long()
            cl2_pred_smt = m.eval(cl2, model_completion=True).as_long()
            cl2_pred = clf2.predict([data])[0]
            cl1_pred = clf1.predict([data])[0]

            if cl1_pred != cl2_pred:
                if cl1_pred_smt != cl1_pred or cl2_pred_smt != cl2_pred:
                    misclassifiedSolverDisagreements.append(
                        (cl1_pred_smt, cl2_pred_smt)
                    )
                else:
                    correctSolverDisagreements.append((cl1_pred_smt, cl2_pred_smt))
                solverDisagreements.append((cl2_pred, cl1_pred))
            else:
                spuriousSolverDisagreements.append((cl1_pred_smt, cl2_pred_smt))

        print("Number of disagreements by solver:", len(solverDisagreements))
        res.write(f"Number of disagreements by solver: {len(solverDisagreements)}\n")
        print("Spurious disagreements by solver:", len(spuriousSolverDisagreements))
        res.write(
            f"Spurious disagreements by solver: {len(spuriousSolverDisagreements)}\n"
        )
        print(
            "Misclassified disagreements by solver:",
            len(misclassifiedSolverDisagreements),
        )
        res.write(
            f"Misclassified disagreements by solver: {len(misclassifiedSolverDisagreements)}\n"
        )
        print("Correct disagreements by solver:", len(correctSolverDisagreements))
        res.write(
            f"Correct disagreements by solver: {len(correctSolverDisagreements)}\n"
        )

    else:
        print("The two classifiers agree on all models")
        res.write("The two classifiers agree on all models\n")

    elapsed_time = time.time() - start_time
    print(f"Time to find disagreements: {elapsed_time}")
    res.write(f"Time to find disagreements: {elapsed_time}\n")

    # Visualize the disagreements
    if VIZ_FLAG:
        # zip the two classifiers predictions
        dt_preds = clf1.predict(X)
        svm_preds = clf2.predict(X)
        zip_preds = [
            (dt_pred, svm_pred) for dt_pred, svm_pred in zip(dt_preds, svm_preds)
        ]

        # filter out the disagreements
        disagreements = [x for x in zip_preds if x[0] != x[1]]
        unique_disagreements = set(disagreements)
        # Visualize the disagreements
        set1_x, set1_y = zip(*solverDisagreements) if solverDisagreements else ([], [])
        set2_x, set2_y = zip(*zip_preds) if zip_preds else ([], [])
        set3_x, set3_y = (
            zip(*misclassifiedSolverDisagreements)
            if misclassifiedSolverDisagreements
            else ([], [])
        )
        set4_x, set4_y = (
            zip(*correctSolverDisagreements) if correctSolverDisagreements else ([], [])
        )
        visualize_disagreements(
            set1_x, set1_y, set2_x, set2_y, set3_x, set3_y, set4_x, set4_y
        )

    print("Results saved to:", res.name)
    res.close()


if __name__ == "__main__":
    main()
