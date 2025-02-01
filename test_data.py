# open file curr_dataset.csv
# read lines, which are file names
# exclude the prefix directoreis of each file name
# remove everything following the LAST "_" char (inlcuding the _ itself)
# plot the frequency of each of remaning stringes as bar chart

import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def get_filename_no_ext(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def get_class_names_by_label(files):
    class_names = defaultdict(int)
    for file in files:
        class_name = get_filename_no_ext(file).rsplit('_', 1)[0]
        class_names[class_name] += 1
    return class_names

def plot_class_frequencies(files):
    class_names = get_class_names_by_label(files)
    plt.bar(class_names.keys(), class_names.values())
    print("Num unique class names:", len(class_names))
    plt.show()


if __name__ == "__main__":
    with open("curr_dataset.csv", "r") as f:
        files = f.readlines()
    files = [file.strip() for file in files]
    plot_class_frequencies(files)