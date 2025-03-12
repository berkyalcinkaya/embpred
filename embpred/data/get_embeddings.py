from itertools import count
from multiprocessing import process
from platform import architecture
import random
from IPython import embed
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
from embpred.features import create_embedding, get_resnet50_model_embedding_model
from skimage.io import imread 


if __name__ == '__main__':
    device = get_device()
    cpu = torch.device('cpu')
    resnet_model = get_resnet50_model_embedding_model(device)

    KFOLDS = 2
    MAPPING_PATH = RAW_DATA_DIR / "mappings.json"
    model_dir = MODELS_DIR / "log_reg"
    temporal_map = TEMPORAL_MAP_PATH
    with open(temporal_map, 'r') as f:
        temporal_map = json.load(f)
    device = get_device()
    dataset1 = PROCESSED_DATA_DIR / "all-classes_carson-224-3depths-noCrop.csv"
    dataset2 = PROCESSED_DATA_DIR / "all-classes_carson-224-3depths.csv"
    output_dir = PROCESSED_DATA_DIR 

    for dataset, name in zip([dataset1, dataset2, dataset2], ["noCrop", "Crop" "cropSingleDepth"]):
        files, labels = get_data_from_dataset_csv(dataset)        
        for file in files:
            assert(os.path.basename(file) in temporal_map)
        embeddings = []
        embeddings_with_temporal = []
        fnames = []
        logger.info(f"Creating embeddings for {name} dataset")
        for file in tqdm(files):
            image = imread(file)
            if name == "cropSingleDepth":
                image = image[:,:,1]
                image = np.stack([image, image, image], axis=2)
                assert image.shape == (224, 224, 3)
            embedding = create_embedding(image, device, resnet_model)
            embedding = embedding.to(cpu)
            embeddings.append(embedding)
            fname = os.path.basename(file)
            fnames.append(fname)    

            temporal_val = temporal_map[fname]
            # temporal value is a scalar, so we need to convert it to a tensor
            temporal_val_tensor = torch.tensor(temporal_val, dtype=torch.float, device=cpu)
            print(temporal_val, temporal_val_tensor, temporal_val_tensor.shape, embedding.shape)

            embedding_with_temporal = torch.cat([embedding, torch.tensor(temporal_map[fname], dtype=torch.float, device=cpu)])
            embeddings_with_temporal.append(embedding_with_temporal)
        embeddings = torch.stack(embeddings)
        embeddings_with_temporal = torch.stack(embeddings_with_temporal)
        
        # Convert tensors to numpy arrays and save embeddings, embeddings_with_temporal,
        # labels, and fnames to a single numpy file (.npz)
        save_path = output_dir / f"resnet50_embeddings_all-class_{name}.npz"
        np.savez(save_path,
                 embeddings=embeddings.cpu().numpy(),
                 embeddings_with_temporal=embeddings_with_temporal.cpu().numpy(),
                 labels=np.array(labels),
                 fnames=np.array(fnames))

        logger.info(f"Saved embeddings to {save_path}")
        # print the size of the npz file
        logger.info(f"Size of the npz file: {os.path.getsize(save_path) / 1e6} MB")
