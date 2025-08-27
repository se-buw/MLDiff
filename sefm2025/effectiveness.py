import os
import numpy as np


# Function to extract information from file names
def extract_info(filename):
    parts = filename.split("_")
    dataset = parts[0]
    classifier1 = parts[1]
    classifier2 = parts[2]
    return dataset, classifier1, classifier2


def disargeements(file_path):
    num_disagreements = None
    spurious_disagreement = None
    misclassified_disagreement = None
    correct_disagreement = None
    time_to_find_disagreements = None
    with open(file_path, "r") as file:
        for line in file:
            if "Number of disagreements by solver" in line:
                num_disagreements = (
                    line.split("Number of disagreements by solver: ")[1]
                ).strip()
            elif "Spurious disagreements by solver" in line:
                spurious_disagreement = (
                    line.split("Spurious disagreements by solver: ")[1]
                ).strip()
            elif "Misclassified disagreements by solver" in line:
                misclassified_disagreement = (
                    line.split("Misclassified disagreements by solver: ")[1]
                ).strip()
            elif "Correct disagreements by solver" in line:
                correct_disagreement = (
                    line.split("Correct disagreements by solver: ")[1]
                ).strip()
            elif "Time to find disagreements" in line:
                time_to_find_disagreements = (
                    line.split("Time to find disagreements: ")[1]
                ).strip()

    return (
        num_disagreements,
        spurious_disagreement,
        misclassified_disagreement,
        correct_disagreement,
        time_to_find_disagreements,
    )


time_data = {}
effectiveness_data = {}
spurious_disagreement_data = {}


def compare_folders(folder1, folder2):
    folder1_files = os.listdir(folder1)
    folder2_files = os.listdir(folder2)

    for file_name in folder1_files:
        if file_name in folder2_files:
            file1_path = os.path.join(folder1, file_name)
            file2_path = os.path.join(folder2, file_name)

            (
                rq2_num_disagreements,
                rq2_spurious_disagreement,
                rq2_misclassified_disagreement,
                rq2_correct_disagreement,
                rq2_time_to_find_disagreements,
            ) = disargeements(file1_path)
            (
                rq4_num_disagreements,
                rq4_spurious_disagreement,
                rq4_misclassified_disagreement,
                rq4_correct_disagreement,
                rq4_time_to_find_disagreements,
            ) = disargeements(file2_path)

            print(f"Disagreements for file '{file_name}':")
            # print(f"Number of disagreements by solver: {rq2_num_disagreements} - {rq4_num_disagreements}")
            # print(f"Spurious disagreements by solver: {(rq2_spurious_disagreement , rq4_spurious_disagreement),rq2_spurious_disagreement}")
            # print(f"Misclassified disagreements by solver: {rq2_misclassified_disagreement}-{rq4_misclassified_disagreement}")
            # print(f"Correct disagreements by solver: {rq2_correct_disagreement}-{rq4_correct_disagreement}")
            dataset, classifier1, classifier2 = extract_info(file_name)
            key = (classifier1, classifier2)
            if key not in effectiveness_data:
                effectiveness_data[key] = {}
            try:
                effectiveness = (
                    float(rq2_spurious_disagreement) - float(rq4_spurious_disagreement)
                ) / float(rq2_spurious_disagreement)
                effectiveness_data[key][dataset] = effectiveness
                # print(f"Effectiveness: {effectiveness}")
            except:
                effectiveness_data[key][dataset] = "-"

            if key not in time_data:
                time_data[key] = {}
            time_data[key][dataset] = float(rq4_time_to_find_disagreements) / float(
                rq2_time_to_find_disagreements
            )
            # print(f"Time to find disagreements: {float(rq4_time_to_find_disagreements)/float(rq2_time_to_find_disagreements)}")
            print()
            if key not in spurious_disagreement_data:
                spurious_disagreement_data[key] = {}
            try:
                spurious_disagreement_data[key][dataset] = rq2_spurious_disagreement
                # print(f"Spurious disagreement: {spurious_disagreement}")
            except:
                spurious_disagreement_data[key][dataset] = "-"


def write_to_file(file_name, data):
    with open(file_name, "w") as file:
        for key, value in data.items():
            model_name = "-".join(key)
            file.write(f"{model_name}: ")
            for dataset in [
                "iris",
                "digits",
                "olivetti",
                "cancer",
            ]:  # Specify the order explicitly for overleaf
                time = value.get(
                    dataset, "N/A"
                )  # Get the time or 'N/A' if dataset not found
                if time != "-":
                    print(f"{dataset}: {time:.2f}  ", end="")
                    file.write(f"{time:.1f} & ")
                else:
                    print(f"{dataset}: {time}  ", end="")
                    file.write(f"{time} & ")
            print()
            file.write("\n")


def write_to_file_spurious(file_name, data):
    with open(file_name, "w") as file:
        for key, value in data.items():
            model_name = "-".join(key)
            file.write(f"{model_name}: ")
            for dataset in [
                "iris",
                "digits",
                "olivetti",
                "cancer",
            ]:  # Specify the order explicitly for overleaf
                time = value.get(
                    dataset, "N/A"
                )  # Get the time or 'N/A' if dataset not found
                if time != "-":
                    print(f"{dataset}: {time}  ", end="")
                    file.write(f"{time} & ")
                else:
                    print(f"{dataset}: {time}  ", end="")
                    file.write(f"{time} & ")
            print()
            file.write("\n")


# Example usage:
folder1 = "sefm2025/results/RQ2"
folder2 = "sefm2025/results/RQ4"
compare_folders(folder1, folder2)

# print(effectiveness_data)
for key in list(effectiveness_data.keys()):
    if key[0] == key[1]:
        del effectiveness_data[key]

values = [
    value
    for pair in effectiveness_data.values()
    for value in pair.values()
    if isinstance(value, (int, float))
]
print(values)

quartiles = np.percentile(values, [25, 50, 75])

print("First Quartile (Q1):", quartiles[0])
print("Second Quartile (Q2) or Median:", quartiles[1])
print("Third Quartile (Q3):", quartiles[2])

count_1 = sum(
    value == 1.0
    for pair in effectiveness_data.values()
    for value in pair.values()
    if isinstance(value, (int, float))
)

print("Count of occurrences of 1.0:", count_1 / len(values))

print(spurious_disagreement_data)

write_to_file_spurious("sefm2025/results/spurious.txt", spurious_disagreement_data)
