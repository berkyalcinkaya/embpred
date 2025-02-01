from torchvision.transforms import v2, ToTensor, Lambda
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
import numpy as np
from PIL import Image
import json
from embpred.config import RAW_DATA_DIR
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image
import skimage 
import tqdm
import glob
from embpred.data.my_transforms import ShuffleColor
plt.rcParams["savefig.bbox"] = 'tight'
from torchvision import transforms as v2

def get_embryo_names_by_from_files(files, labels):
    embryo_names_to_files = {}
    embryo_names_to_count = {}
    embryo_names_to_labels = {}

    for file, label in zip(files, labels):
        embryo_name = get_filename_no_ext(file).rsplit('_', 1)[0]

        if embryo_name not in embryo_names_to_files:
            embryo_names_to_files[embryo_name] = []
        embryo_names_to_files[embryo_name].append(file)

        if embryo_name not in embryo_names_to_count:
            embryo_names_to_count[embryo_name] = 0
        embryo_names_to_count[embryo_name] += 1

        if embryo_name not in embryo_names_to_labels:
            embryo_names_to_labels[embryo_name] = []
        embryo_names_to_labels[embryo_name].append(label)

    return embryo_names_to_files, embryo_names_to_count, embryo_names_to_labels


def get_class_names_by_label(mapping_dict)->dict:
    label_ints = mapping_dict.values()
    class_name_dict = {}
    for label_int in label_ints:
        class_name_dict[label_int] = "-".join([key for key, value in mapping_dict.items() if value == label_int])
    return class_name_dict

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

def kfold_split(strings, n_splits=5, random_state=None, test_size=0.25):
    """
    Perform k-fold split on a list of strings.

    Parameters:
    strings (list): List of strings to be split.
    n_splits (int): Number of folds. Default is 5.
    random_state (int or None): Random state for reproducibility. Default is None.

    Returns:
    List of tuples: Each tuple contains two lists (train_strings, test_strings) for each fold.
                    If n_splits < 2, returns a single train-test split.
    """
    # Ensure we have a non-empty list of strings
    assert len(strings) > 0, "The list of strings must not be empty."
    
    if n_splits < 2:
        # Perform a single train-test split
        train_strings, test_strings = train_test_split(
            strings, test_size=test_size, random_state=random_state
        )
        return [(train_strings, test_strings)]
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # List to store train-test splits
    splits = []
    
    # Perform the splits
    for train_index, test_index in kf.split(strings):
        train_strings = [strings[i] for i in train_index]
        test_strings = [strings[i] for i in test_index]
        splits.append((train_strings, test_strings))
    
    return splits


def get_transforms(image_net_transforms = False):
    if image_net_transforms:
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([ShuffleColor()], p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomApply([ShuffleColor()], p=0.5)
        ])
    
    return transforms

def get_basic_transforms():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
    ])


class CustomImageDataset(Dataset):
    def __init__(self, img_paths, img_labels, img_transform=None, encode_labels = True, num_channels=1, channel_idx=None, do_normalize=True, check_exists=True):
        self.img_paths = img_paths
        if check_exists:
            num_exists = 0
            for img_path in self.img_paths:
                if os.path.exists(img_path):
                    num_exists += 1
            if num_exists < len(self.img_paths):
                raise ValueError(f"Only {num_exists} of {len(self.img_paths)} images exist.")
        print(f"--All images exist: {num_exists} of {len(self.img_paths)} images exist.--")  
        
        # determine image type from the first image in img_paths and save this as an attribute
        self.img_type = self.img_paths[0].split(".")[-1]

        self.img_labels = img_labels
        self.num_classes = len(np.unique(self.img_labels))
        self.transform = img_transform
        self.encode_labels = encode_labels
        
        self.num_channels=num_channels
        self.channel_idx = channel_idx
        self.do_normalize = do_normalize
        
        # Define the label transformation function once
        if self.encode_labels:
            self.label_transform = Lambda(lambda y: 
                torch.zeros(self.num_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        else:
            self.label_transform = None

    def __len__(self):
        return len(self.img_labels)
    
    def load_image(self, im_file):
        if self.num_channels not in [1,3]:
            im_npy = skimage.io.imread(im_file)
            im = torch.from_numpy(im_npy)
        else:
            im = read_image(im_file)
        
        if self.channel_idx is not None:
            im = im[self.channel_idx, :, :]
        
        if self.do_normalize:
            im = im.float() / 255

        return im
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]
        image = self.load_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.encode_labels:
            label = self.label_transform(label)
        
        return image, label
    
    def get_labels(self):
        return self.img_labels
    
    def get_num_classes(self):
        return self.num_classes