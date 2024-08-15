from itertools import count
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
from embpred.modeling.models import FirstNet, count_parameters
from embpred.dataset import (transforms, CustomImageDataset, get_data_from_dataset_csv, 
                            get_filename_no_ext, stratified_kfold_split, load_mappings, get_class_names_by_label)
from embpred.modeling.train_utils import get_device, train_and_evaluate, evaluate, configure_model_dir


def build_model(model_name, num_classes):
    return FirstNet(num_classes=num_classes)

if __name__ == "__main__":
    MODELS = ["FirstNet"]
    KFOLDS = 2
    EPOCHS = 50
    LR = 0.001
    WEIGHT_DECAY = 0.0001
    BATCH_SIZE = 32

    mappings = load_mappings()
    device = get_device()
    datasets = glob(str(PROCESSED_DATA_DIR / "*.csv"))
    for do_sampling in [True, False]:
        for model_name in MODELS:
            for dataset in datasets:
                files, labels = get_data_from_dataset_csv(dataset)
                dataset, num_classes = get_filename_no_ext(dataset), len(np.unique(labels))
                logger.info(f"DATASET {dataset} | NUM CLASSES: {num_classes}")

                additional_ids = ["sampled"] if do_sampling else ["unsampled"]

                mapping = mappings[dataset]
                class_names_by_label = get_class_names_by_label(mapping)
                logger.info(class_names_by_label)
                model_dir = configure_model_dir(model_name, dataset, mapping, 
                                                additional_ids=additional_ids)
                logger.info(f"MODEL DIR: {model_dir}")
                
                accs, aucs, macros,  = [], [], []
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

                    train_loader = DataLoader(train_data, batch_size=64, sampler=sampler, shuffle=do_shuffle, num_workers=4)
                    val_loader = DataLoader(val_data, batch_size=64, num_workers=4)

                    model = build_model(model_name, num_classes=train_data.get_num_classes())
                    param_count = count_parameters(model)
                    logger.info(f"{str(model)}")
                    logger.info(f"Model parameters: {param_count}")
                    model.to(device)
                    
                    logger.info(f"Loaded model to {device}")
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                    criterion = nn.CrossEntropyLoss()
                    
                    model_save_path = os.path.join(log_dir, "model.pth")
                    logger.info(f"BEST WEIGHTS: {model_save_path}")
            
                    val_micro, val_auc, val_macro = train_and_evaluate(model, train_loader, val_loader, 
                                                                       optimizer, device, criterion, False,5,
                                                                       EPOCHS, writer, best_model_path=model_save_path, 
                                                                       class_names=class_names_by_label)
                    val_micro, val_auc, val_macro, _ = evaluate(model, device, val_loader)
                    logger.info(f'(Initial Performance Last Epoch) | test_micro={(val_micro * 100):.2f}, '
                                    f'test_macro={(val_macro * 100):.2f}, test_auc={(val_auc * 100):.2f}')
                    accs.append(val_micro)
                    aucs.append(val_auc)
                    macros.append(val_macro)

                result_str = f'(K Fold Final Result)| avg_acc={(np.mean(accs) * 100):.2f} +- {(np.std(accs) * 100): .2f}, ' \
                 f'avg_auc={(np.mean(aucs) * 100):.2f} +- {np.std(aucs) * 100:.2f}, ' \
                 f'avg_macro={(np.mean(macros) * 100):.2f} +- {np.std(macros) * 100:.2f}\n'
                logger.info(result_str)


