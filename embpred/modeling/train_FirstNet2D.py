from itertools import count
from platform import architecture
from loguru import logger
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
from embpred.config import MODELS_DIR, PROCESSED_DATA_DIR
from embpred.modeling.models import FirstNet2D, count_parameters, SimpleNet3D, CustomResNet18, CustomResNet50
from embpred.data.dataset import (transforms, CustomImageDataset, get_data_from_dataset_csv, 
                            get_filename_no_ext, stratified_kfold_split, load_mappings, get_class_names_by_label)
from embpred.modeling.train_utils import get_device, train_and_evaluate, evaluate, configure_model_dir
from embpred.modeling.utils import report_kfolds_results
import csv


model_mappings =  {
    "SimpleNet3D": SimpleNet3D,
    "CustomResNet18": CustomResNet18,
    "CustomResNet50": CustomResNet50 
}

if __name__ == "__main__":
    # Define the models to train with 
    MODELS = [
        ("SimpleNet3D", SimpleNet3D, {}),
        ("CustomResNet18-1layer", CustomResNet18, {"num_dense_layers": 1, "dense_neurons": 64}),
        ("CustomResNet18-2layer", CustomResNet18, {"num_dense_layers": 2, "dense_neurons": 64})
        ("CustomResNet50-1layer", CustomResNet50, {"num_dense_layers": 1, "dense_neurons": 64})
    ]

    KFOLDS = 4
    EPOCHS = 30
    LR = 0.001
    WEIGHT_DECAY = 0.0001
    BATCH_SIZE = 32

    mappings = load_mappings()
    device = get_device()
    datasets = glob(str(PROCESSED_DATA_DIR / "*.csv"))
    for do_sampling in [False, True]:
        for model_name, model_class, architecture_params in MODELS:
            for dataset in datasets:
                files, labels = get_data_from_dataset_csv(dataset)
                dataset, num_classes = get_filename_no_ext(dataset), len(np.unique(labels))
                logger.info(f"DATASET {dataset} | NUM CLASSES: {num_classes}")

                additional_ids = ["sampled"] if do_sampling else ["unsampled"]

                mapping = mappings[dataset]
                class_names_by_label = get_class_names_by_label(mapping)
                logger.info(class_names_by_label)
                model_dir = configure_model_dir(model_name, dataset, mapping, architecture=architecture_params,
                                                additional_ids=additional_ids)
                logger.info(f"MODEL DIR: {model_dir}")
                
                accs, aucs, macros, losses = [], [], [], []
                conf_mats = np.zeros((num_classes, num_classes))
                k_fold_splits = stratified_kfold_split(files, labels, n_splits=KFOLDS)
                for idx, fold in enumerate(k_fold_splits):
                    logger.info(f"Fold {idx+1}/{KFOLDS}")
                    log_dir = f"{model_dir}/runs/fold_{idx}"
                    logger.info(f"Tensorboard output >> {log_dir}")
                    writer = SummaryWriter(log_dir=log_dir)
                    train_ims, train_labels, val_ims, val_labels = fold

                    train_data = CustomImageDataset(train_ims, train_labels, img_transform=transforms)
                    val_data = CustomImageDataset(val_ims, val_labels, img_transform=transforms)

                    sampler, do_shuffle = (ImbalancedDatasetSampler(train_data), False) if do_sampling else (None, True)

                    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, shuffle=do_shuffle, num_workers=4)
                    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=4)

                    model = model_class(num_classes=train_data.get_num_classes(), **architecture_params)
                    param_count = count_parameters(model)
                    logger.info(f"{str(model)}")
                    logger.info(f"Model parameters: {param_count}")
                    model.to(device)
                    
                    logger.info(f"Loaded model to {device}")
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                    criterion = nn.CrossEntropyLoss()
                    
                    model_save_path = os.path.join(log_dir, "model.pth")
                    logger.info(f"BEST WEIGHTS: {model_save_path}")
            
                    val_micro, val_auc, val_macro, val_losses = train_and_evaluate(model, train_loader, val_loader, 
                                                                       optimizer, device, criterion, False,5,
                                                                       EPOCHS, writer, best_model_path=model_save_path, 
                                                                       class_names=class_names_by_label, early_stop_epochs=20, 
                                                                       do_early_stop=True)
                    val_micro, val_aucs, val_macro, avg_loss, conf_mat = evaluate(model, device, val_loader, loss=criterion, get_conf_mat=True)
                    val_auc=np.mean(val_aucs)
                    
                    logger.info(f'(Initial Performance Last Epoch) | test_loss={avg_loss:.4f} | test_micro={(val_micro * 100):.2f}, '
                                    f'test_macro={(val_macro * 100):.2f}, test_auc={(val_auc * 100):.2f}')
                    conf_mats += conf_mat
                    accs.append(val_micro)
                    aucs.append(val_auc)
                    macros.append(val_macro)
                    losses.append(avg_loss)
                
                
                ### END of kFolds: Record model performance
                report_kfolds_results(model_dir, accs, aucs, macros, losses, conf_mats, KFOLDS)
               
                
                

