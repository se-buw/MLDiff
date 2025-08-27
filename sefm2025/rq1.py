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

    with open(file_path, "r") as file:
        for line in file:
            if "Number of unique disagreements" in line:
                num_disagreements = (
                    line.split("Number of unique disagreements: ")[1]
                ).strip()
    return num_disagreements


disagreement_data = {}


def find_disagreements(folder):
    files = os.listdir(folder)
    # folder2_files = os.listdir(folder2)

    for file_name in files:
        file_path = os.path.join(folder, file_name)

        num_disagreements = disargeements(file_path)

        dataset, classifier1, classifier2 = extract_info(file_name)
        key = (classifier1, classifier2)
        if key not in disagreement_data:
            disagreement_data[key] = {}
        try:
            dis = int(num_disagreements)
            disagreement_data[key][dataset] = dis
        except:
            disagreement_data[key][dataset] = "-"


# Example usage:
folder = "sefm2025/results/RQ1"
find_disagreements(folder)

possible_disagreement = {"cancer": 2, "digits": 90, "iris": 6, "olivetti": 90}


ratios = {}

# Iterate over disagreement_data keys
for key in disagreement_data:
    ratio_dict = {}
    for subkey in disagreement_data[key]:
        # ignore cancer dataset
        if subkey == "cancer":
            continue
        ratio_dict[subkey] = (
            disagreement_data[key][subkey] / possible_disagreement[subkey]
        )
        print(key, subkey, ratio_dict[subkey])
    ratios[key] = ratio_dict


# Collect all values from the ratios dictionary
values = []
for ratio_dict in ratios.values():
    values.extend(ratio_dict.values())

# Convert the values list to a numpy array
values_array = np.array(values)

# Calculate quartiles
quartiles = np.percentile(values_array, [25, 50, 75])

# Print quartiles
print("First Quartile (Q1):", quartiles[0])
print("Second Quartile (Q2, Median):", quartiles[1])
print("Third Quartile (Q3):", quartiles[2])
