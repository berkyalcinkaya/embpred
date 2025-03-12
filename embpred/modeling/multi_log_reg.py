from itertools import count
from multiprocessing import process
from platform import architecture
import random
from loguru import logger
from requests import get
import test
from tqdm import tqdm
import numpy as np
#import pandas as pd
import json
from glob import glob
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchsampler import ImbalancedDatasetSampler
from embpred.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_STATE, TEMPORAL_MAP_PATH
from embpred.data.make_dataset import process_embryo
from embpred.modeling.models import (BiggestNet3D224, SmallerNet3D224, count_parameters, SimpleNet3D, CustomResNet18, CustomResNet50, 
                                    BiggerNet3D224, SmallerNet3D224, WNet, BigWNet)
from embpred.data.dataset import (get_basic_transforms, CustomImageDataset, get_data_from_dataset_csv, 
                            get_filename_no_ext, process_embryo_split, stratified_kfold_split, kfold_split,
                            load_mappings, get_class_names_by_label, 
                            get_transforms, get_embryo_names_by_from_files)
from embpred.data.balance import DataBalancer
from embpred.modeling.train_utils import get_device, train_and_evaluate, evaluate, configure_model_dir
from embpred.modeling.utils import report_kfolds_results
from embpred.modeling.loss import get_class_weights, weighted_cross_entropy_loss
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from embpred.features import create_resnet50_embedding


if __name__ == '__main__':

    KFOLDS = 2
    MAPPING_PATH = RAW_DATA_DIR / "mappings.json"
    model_dir = MODELS_DIR / "log_reg"
    device = get_device()
    dataset = PROCESSED_DATA_DIR / "all-classes_carson-224-3depths-noCrop.csv"
    files, labels = get_data_from_dataset_csv(dataset)

    with open(TEMPORAL_MAP_PATH, 'r') as f:
        temporal_map = json.load(f)
    
    print(len(files))
    print(len(temporal_map))
    
    for file in files:
        print(os.path.basename(file))
        assert(os.path.basename(file) in temporal_map)

    # embryo_names_to_files, embryo_names_to_count, embryo_names_to_labels = get_embryo_names_by_from_files(files, labels)
    # k_fold_splits_by_embryo = kfold_split(list(embryo_names_to_files.keys()), n_splits=KFOLDS, random_state=RANDOM_STATE, val_size=0.1,
    #                                       test_size=0.1)
    # train_embryos, val_embryos, test_embryos = k_fold_splits_by_embryo[0]
    # split = process_embryo_split( embryo_names_to_files, embryo_names_to_labels, train_embryos, val_embryos, test_embryos,
    #                              merge_train_val=True)
    # X_train, y_train, X_test, y_test = split

    # # one hot encode labels
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)


    # model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # print("Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))