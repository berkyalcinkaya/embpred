from itertools import count
from multiprocessing import process
from platform import architecture
from pyexpat import model
import random
from venv import create
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
from embpred.config import DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_STATE, TEMPORAL_MAP_PATH
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
from embpred.features import create_embedding, get_resnet50_model_embedding_model
from skimage.io import imread 
from embpred.modeling.train_utils import write_data_split
from sklearn.decomposition import PCA

def train_log_reg(npz_file, weights=None, MAX_ITER=100):
    """"
    Trains a logistic regression model on the embeddings from the given npz file."
    """

    model_name = f"log_reg_{npz_file.split('.')[0]}"
    if weights:
        model_name += "_weighted"
    model_dir = MODELS_DIR / model_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    dataset = DATA_DIR / npz_file
    data = np.load(dataset)

    dataset = DATA_DIR / "resnet50_emb_noCrop.npz"
    data = np.load(dataset)
    embeds = data["embeddings"]
    labels = data["labels"]
    files = data["fnames"]

    assert(len(np.unique(files)) == len(files))

    
    # convert files to a list of strings
    files = [str(f) for f in files]

    embryo_names_to_files, embryo_names_to_count, embryo_names_to_labels = get_embryo_names_by_from_files(files, labels)
    #print(embryo_names_to_files)
    k_fold_splits_by_embryo = kfold_split(list(embryo_names_to_files.keys()), n_splits=1, random_state=RANDOM_STATE, val_size=0.1,
                                           test_size=0.2)
    train_embryos, val_embryos, test_embryos = k_fold_splits_by_embryo[0]

    logger.info(f"Number of train embryos: {len(train_embryos)}")
    logger.info(f"Number of val embryos: {len(val_embryos)}")
    logger.info(f"Number of test embryos: {len(test_embryos)}")

    # write the train, val, and test embryo names to a file
    write_data_split(train_embryos, val_embryos, test_embryos, loc = model_dir)

    split = process_embryo_split(embryo_names_to_files, embryo_names_to_labels, train_embryos, val_embryos, test_embryos,
                                  merge_train_val=True)
    X_train_files, y_train, X_test_files, y_test = split
    
    # log the sizes of the splits
    logger.info("Train size: {}".format(len(X_train_files)))
    logger.info("Val size: {}".format(len(X_test_files)))
    logger.info(f"Train Percentage: {len(X_train_files) / (len(X_train_files) + len(X_test_files))}")

    X_train = []
    X_test = []
    for file in tqdm(X_train_files):
        X_train.append(embeds[files.index(file)])
    for file in tqdm(X_test_files):
        X_test.append(embeds[files.index(file)])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=MAX_ITER, verbose=0, class_weight = weights) #class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"--------{model_name}-------")
    print("Test Accuracy: {:.2f}%".format(test_acc * 100))
    print("Classification Report:")
    print(report)
    print()

    # Write the results to a file
    with open(model_dir / "log_reg_results.txt", "w") as f:
        f.write("Test Accuracy: {:.2f}%\n".format(test_acc * 100))
        f.write("Classification Report:\n")
        f.write(report)
    return model_name, test_acc


def main():
    npz_files = ["resnet50_emb_crop.npz", "resnet50_emb_noCrop.npz", "resnet50_emb_cropSingleDepth.npz"]
    for npz_file in npz_files:
        assert(os.path.exists(DATA_DIR / npz_file))
    model_acc = []
    for npz_file in npz_files:
        for weight in [None, "balanced"]:
            model_name, acc = train_log_reg(npz_file, weight, MAX_ITER=2000)
            model_acc.append((model_name, acc))
            print(model_name, acc)
    print(model_acc)

    # find best model
    best_model = max(model_acc, key=lambda x: x[1])
    print(f"Best Model: {best_model[0]}")
    print(f"Best Model Accuracy: {best_model[1]}")

if __name__ == '__main__':
    main()