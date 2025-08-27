import itertools

datasets = ["iris", "digits", "olivetti", "cancer"]
classifiers = ["dt", "svm", "logreg", "mlp"]
hidden_layer_sizes = [(10,), (20, 10, 5)]
epsilon_decision_distance = 0
epsilon_argmax = 0
save_dir = "RQ1"
feature_constraints = False

with open("sefm2025/scripts/rq1.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for dataset, classifier1, classifier2 in itertools.product(
        datasets, classifiers, classifiers
    ):
        if classifier1 == classifier2:
            continue
        print(
            f"python -m mldiff.diff -d={dataset} -c1={classifier1} -c2={classifier2} {'-pca' if dataset == 'olivetti' else ''} -ed {epsilon_decision_distance} -ea {epsilon_argmax} {'-fc' if feature_constraints else '' } -sd={save_dir} "
        )
        f.write(
            f"timeout -k 0s 60m python -m mldiff.diff -d={dataset} -c1={classifier1} -c2={classifier2} {'-pca' if dataset == 'olivetti' else ''} -ed {epsilon_decision_distance} -ea {epsilon_argmax} {'-fc' if feature_constraints else '' } -sd={save_dir} -v \n"
        )
