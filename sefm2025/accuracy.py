"""
This script parses accuracy results from text files and aggregates them.
"""
import os


def parse_file(file_path):
    with open(file_path, "r") as file:
        content = file.readlines()

    dataset = None
    classifier = None
    accuracy = None
    f1_score = None

    for line in content:
        line = line.strip()
        if line.startswith("Dataset:"):
            dataset = line.split(": ")[1]
        elif "Classifier" in line and "accuracy" in line:
            classifier = line.split(" ")[1]
            accuracy = float(line.split(": ")[1])
        elif "Classifier" in line and "f1-score" in line:
            f1_score = float(line.split(": ")[1])

    return dataset, classifier, accuracy, f1_score


def main(folder_path):
    acc = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            dataset, classifier, accuracy, f1_score = parse_file(file_path)
            acc.append((classifier, dataset, accuracy, f1_score))

    unique_data = set(str(item) for item in acc)

    # convert the unique elements back to tuples:
    unique_tuples = [eval(item) for item in unique_data]

    # sort by classifier
    unique_tuples.sort(key=lambda x: x[0])

    with open("sefm2025/results/accuracy.txt", "w") as file:
        for item in unique_tuples:
            print(f"{item[0]} - {item[1]}: {item[2]*100:.2f}\%, {item[3]*100:.2f}\%")
            file.write(f"{item[0]} - {item[1]}: {item[2]*100:.2f}, {item[3]*100:.2f}\n")


if __name__ == "__main__":
    folder_path = "sefm2025/results/RQ2"
    main(folder_path)
