import os


# Function to extract information from file names
def extract_info(filename):
    parts = filename.split("_")
    dataset = parts[0]
    classifier1 = parts[1]
    classifier2 = parts[2]
    return dataset, classifier1, classifier2


def disargeements(file_path):
    time_to_find_disagreements = None
    with open(file_path, "r") as file:
        for line in file:
            if "Time to find disagreements" in line:
                time_to_find_disagreements = (
                    line.split("Time to find disagreements: ")[1]
                ).strip()

    return time_to_find_disagreements


time_data = {}


def compare_folders(folder1, folder2):
    folder1_files = os.listdir(folder1)
    folder2_files = os.listdir(folder2)

    for file_name in folder1_files:
        if file_name in folder2_files:
            file1_path = os.path.join(folder1, file_name)
            file2_path = os.path.join(folder2, file_name)

            rq2_time_to_find_disagreements = disargeements(file1_path)
            rq4_time_to_find_disagreements = disargeements(file2_path)

            dataset, classifier1, classifier2 = extract_info(file_name)
            key = (classifier1, classifier2)

            if key not in time_data:
                time_data[key] = {}
            if rq4_time_to_find_disagreements is None:
                time_data[key][dataset] = "x"
            else:
                time_data[key][dataset] = float(rq4_time_to_find_disagreements) / float(
                    rq2_time_to_find_disagreements
                )
            # print()


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
                if time != "x":
                    print(f"{dataset}: {time:.2f}  ", end="")
                    file.write(f"{time:.2f} & ")
                else:
                    print(f"{dataset}: {time}  ", end="")
                    file.write(f"{time} & ")
            print()
            file.write("\n")


folder1 = "sefm2025/results/RQ2"
folder2 = "sefm2025/results/RQ3"
compare_folders(folder1, folder2)


write_to_file("sefm2025/results/time_ratio_between_rq2-3.txt", time_data)
