import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
plt.rcParams["savefig.bbox"] = 'tight'
from torchvision.transforms import v2, ToTensor, Lambda
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from PIL import Image
import json
from embpred.config import RAW_DATA_DIR

def load_mappings(pth = "EmbStages/dataset1/mappings.json"):
    mapping_path =  RAW_DATA_DIR / pth
    with open(mapping_path, "r") as f:
        return json.load(f)

def get_filename_no_ext(filepath):
    """
    Returns the filename without its extension and preceding directories.

    Parameters:
    - filepath: The full path to the file.

    Returns:
    - filename_no_ext: The name of the file without its extension.
    """
    basename = os.path.basename(filepath)
    filename_no_ext = os.path.splitext(basename)[0]
    return filename_no_ext

def get_data_from_dataset_csv(pth):
    df = pd.read_csv(pth)
    return df["path"].tolist(), df["label"].to_numpy()

def stratified_kfold_split(image_paths, labels, n_splits=5, random_state=None, test_size = 0.25):
    """
    Perform stratified k-fold split on lists of image paths and labels.

    Parameters:
    image_paths (list): List of image paths.
    labels (list): List of labels corresponding to the image paths.
    n_splits (int): Number of folds. Default is 5.
    random_state (int or None): Random state for reproducibility. Default is None.

    Returns:
    List of tuples: Each tuple contains four lists (train_image_paths, train_labels, test_image_paths, test_labels)
                    for each fold. If n_splits < 2, returns a single train-test split.
    """
    # Ensure we have the same number of image paths and labels
    assert len(image_paths) == len(labels), "The length of image paths and labels must be the same."
    
    if n_splits < 2:
        # Perform a single train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
        )
        return [(X_train, y_train, X_test, y_test)]
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # List to store train-test splits
    splits = []
    
    # Perform the splits
    for train_index, test_index in skf.split(image_paths, labels):
        X_train = [image_paths[i] for i in train_index]
        y_train = [labels[i] for i in train_index]
        X_test = [image_paths[i] for i in test_index]
        y_test = [labels[i] for i in test_index]
        splits.append((X_train, y_train, X_test, y_test))
    
    return splits


# rotation
transforms = v2.Compose([
        v2.Resize((800, 800), interpolation=Image.BICUBIC),
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomVerticalFlip(p=0.5),
      v2.RandomRotation(degrees=(0, 180))
])

class CustomImageDataset(Dataset):
    def __init__(self, img_paths, img_labels, img_transform=None, encode_labels = True):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.num_classes = len(np.unique(self.img_labels))
        self.transform = img_transform
        self.encode_labels = encode_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]
        image = read_image(img_path)/255
        if self.transform:
            image = self.transform(image)
        if self.encode_labels:
            transform = Lambda(lambda y: 
                               torch.zeros(self.num_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
            label = transform(label)
        
        return image, label
    
    def get_labels(self):
        return self.img_labels
    
    def get_num_classes(self):
        return self.num_classes
