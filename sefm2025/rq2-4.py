import os
from prettytable import PrettyTable

# Function to extract information from file names
def extract_info(filename):
    parts = filename.split('_')
    dataset = parts[0]
    classifier1 = parts[1]
    classifier2 = parts[2]  
    return dataset, classifier1, classifier2

# Dictionary to store time data
time_data = {}

# Directory where files are located
folder_path = 'sefm2025/results/RQ3'

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Assuming all files are text files
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("Time to find disagreements:"):
                    time = float(line.split(": ")[1].strip())
                    dataset, classifier1, classifier2 = extract_info(filename)
                    key = (classifier1, classifier2)
                    if key not in time_data:
                        time_data[key] = {}
                    time_data[key][dataset] = time


with open('sefm2025/results/time_rq3.txt', 'w') as file:
    for key, value in time_data.items():
        model_name = '-'.join(key)  # Joining the tuple elements with '-'
        print(f"{model_name}: ", end="")
        file.write(f"{model_name}: ")
        for dataset in ['iris', 'digits', 'olivetti', 'cancer']:  # Specify the order explicitly for overleaf
            time = value.get(dataset, 'N/A')  # Get the time or 'N/A' if dataset not found
            if time != 'N/A':
                print(f"{dataset}: {time:.2f}  ", end="")
                file.write(f"{time:.2f} & ")
            else:
                print(f"{dataset}: {time}  ", end="")
                file.write(f"x & ")
        print()  
        file.write("\n")