from itertools import count
from platform import architecture
import random
from loguru import logger
from requests import get
from tqdm import tqdm
import numpy as np
#import pandas as pd
from glob import glob
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchsampler import ImbalancedDatasetSampler
from embpred.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_STATE
from embpred.modeling.models import (BiggestNet3D224, SmallerNet3D224, count_parameters, SimpleNet3D, CustomResNet18, CustomResNet50, 
                                    BiggerNet3D224, SmallerNet3D224, WNet, BigWNet)
from embpred.data.dataset import (get_basic_transforms, CustomImageDataset, get_data_from_dataset_csv, 
                            get_filename_no_ext, stratified_kfold_split, kfold_split,
                            load_mappings, get_class_names_by_label, 
                            get_transforms, get_embryo_names_by_from_files)
from embpred.data.balance import DataBalancer
from embpred.modeling.train_utils import get_device, train_and_evaluate, evaluate, configure_model_dir
from embpred.modeling.utils import report_kfolds_results
from embpred.modeling.loss import get_class_weights, weighted_cross_entropy_loss
import csv


if __name__ == '__main__':

    KFOLDS = 5
    MAPPING_PATH = RAW_DATA_DIR / "mappings.json"
    device = get_device()
    dataset = PROCESSED_DATA_DIR / "all-classes_carson-224-3depths-noCrop.csv"
    files, labels = get_data_from_dataset_csv(dataset)
    embryo_names_to_files, embryo_names_to_count, embryo_names_to_labels = get_embryo_names_by_from_files(files, labels)

    #k_fold_splits = stratified_kfold_split(files, labels, n_splits=KFOLDS)
    k_fold_splits = []
    k_fold_splits_by_embryo = kfold_split(list(embryo_names_to_files.keys()), n_splits=KFOLDS, random_state=RANDOM_STATE)
    for train_embryos, val_embryos in k_fold_splits_by_embryo:
        train_files, train_labels = [], []
        for embryo in train_embryos:
            train_files.extend(embryo_names_to_files[embryo])
            train_labels.extend(embryo_names_to_labels[embryo])
        val_files, val_labels = [], []
        for embryo in val_embryos:
            val_files.extend(embryo_names_to_files[embryo])
            val_labels.extend(embryo_names_to_labels[embryo])
        k_fold_splits.append((train_files, train_labels, val_files, val_labels))