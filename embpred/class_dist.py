import os
import argparse
import matplotlib.pyplot as plt
from collections import Counter

def get_class_distribution(data_dir, file_type=".tif"):
    """
    Returns the class distribution from the data directory. Each subdirectory in data_dir
    is treated as a class, and the number of images (JPEG) in each class is counted.
    
    Args:
        data_dir (str): Path to the directory containing class subdirectories.
    
    Returns:
        class_counts (dict): A dictionary mapping each class to the number of images in that class.
    """
    class_counts = Counter()

    # Loop through each subdirectory (each representing a class)
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):  # Ensure it's a directory
            # Count the number of JPEG files in this class directory
            class_counts[class_dir] = len([f for f in os.listdir(class_path) if f.endswith(file_type)])

    return class_counts

def plot_class_distribution(class_counts):
    """
    Plots a bar chart of the class distribution.
    
    Args:
        class_counts (dict): A dictionary mapping class names to the number of images.
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Utility to generate class distribution for image dataset")
    parser.add_argument('data_dir', type=str, help="Path to the data directory containing class subdirectories")
    args = parser.parse_args()

    # Get the class distribution
    class_counts = get_class_distribution(args.data_dir)

    # Print the class distribution
    for class_name, count in class_counts.items():
        print(f"Class: {class_name}, Images: {count}")

    # Plot the class distribution
    plot_class_distribution(class_counts)

if __name__ == "__main__":
    main()
