from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from embpred.config import FIGURES_DIR, PROCESSED_DATA_DIR
import random
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def display_histograms(image_paths, classes, func = None):
    """
    Randomly samples 4 images of each class and displays their histograms in one Matplotlib figure.

    Parameters:
    image_paths (list): List of paths to images.
    classes (list): Parallel list of classes corresponding to each image.
    """
    # Ensure we have the same number of image paths and classes
    assert len(image_paths) == len(classes), "The length of image paths and classes must be the same."
    
    # Combine image paths and classes into a list of tuples and group by class
    data = list(zip(image_paths, classes))
    class_dict = {}
    
    for path, cls in data:
        if cls not in class_dict:
            class_dict[cls] = []
        class_dict[cls].append(path)
    
    # Set up the plot
    num_classes = len(class_dict)
    fig, axes = plt.subplots(num_classes, 4, figsize=(20, 5 * num_classes))
    fig.tight_layout(pad=5.0)
    
    for i, cls in enumerate(class_dict):
        # Randomly sample 4 images for each class
        sampled_paths = random.sample(class_dict[cls], 4)
        
        for j, path in enumerate(sampled_paths):
            image = io.imread(path)
            flattened_image = image.flatten()
            
            # Plot histogram
            ax = axes[i, j] if num_classes > 1 else axes[j]
            ax.hist(flattened_image, bins=50, edgecolor='black')
            ax.set_title(f'Class {cls} - Image {j+1}')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
    
    plt.show()

def plot_histogram_2d_array(array, bins=10, title='Histogram', xlabel='Value', ylabel='Frequency'):
    """
    Produces and displays a histogram for a given 2D array.

    Parameters:
    array (numpy.ndarray): The input 2D array.
    bins (int): The number of bins to use for the histogram. Default is 10.
    title (str): The title of the histogram. Default is 'Histogram'.
    xlabel (str): The label for the x-axis. Default is 'Value'.
    ylabel (str): The label for the y-axis. Default is 'Frequency'.
    """
    # Flatten the 2D array into a 1D array
    flattened_array = array.flatten()
    
    # Create the histogram
    plt.hist(flattened_array, bins=bins, edgecolor='black')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Display the histogram
    plt.show()

