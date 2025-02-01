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
from embpred.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from embpred.modeling.models import (BiggestNet3D224, SmallerNet3D224, count_parameters, SimpleNet3D, CustomResNet18, CustomResNet50, 
                                    BiggerNet3D224, SmallerNet3D224)
from embpred.data.dataset import (get_basic_transforms, CustomImageDataset, get_data_from_dataset_csv, 
                            get_filename_no_ext, stratified_kfold_split, kfold_split,
                            load_mappings, get_class_names_by_label, 
                            get_transforms, get_embryo_names_by_from_files)
from embpred.data.balance import DataBalancer
from embpred.modeling.train_utils import get_device, train_and_evaluate, evaluate, configure_model_dir
from embpred.modeling.utils import report_kfolds_results
import csv

# write code to ensure that pytorch caches are cleared and that the GPU memory is freed up
torch.cuda.empty_cache()

# TODO: before the split, randomly sample to lower class frequencies (500). Then split inot train test validation

MAPPING_PATH = RAW_DATA_DIR / "mappings.json"

model_mappings =  {
    "SimpleNet3D": SimpleNet3D,
    "CustomResNet18": CustomResNet18,
    "CustomResNet50": CustomResNet50 ,
    "BiggerNet3D224": BiggerNet3D224,
    "SmallerNet3D224": SmallerNet3D224
}

def do_random_sample(PRE_RANDOM_SAMPLE, files, labels):
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    logger.info(f"PRE-SAMPLING: {class_counts}")
    for label, count in class_counts.items():
        if count < PRE_RANDOM_SAMPLE:
            logger.info(f"Class {label} has {count} images, sampling all")
        else:
            logger.info(f"Class {label} has {count} images, sampling {PRE_RANDOM_SAMPLE}")
    sampled_files, sampled_labels = [], []
    for label in np.unique(labels):
        label_files = [f for f, l in zip(files, labels) if l == label]
        sampled_files.extend(np.random.choice(label_files, PRE_RANDOM_SAMPLE, replace=False))
        sampled_labels.extend([label] * PRE_RANDOM_SAMPLE)
    files, labels = sampled_files, sampled_labels
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    logger.info(f"POST-SAMPLING: {class_counts}")
    return files,labels

if __name__ == "__main__":
    # Define the models to train with 
    MODELS = [
        ("BiggerNet3D224-emb-kfolds", BiggerNet3D224, {})
        #("CustomResNet18-1layer-full-balance", CustomResNet18, {"num_dense_layers": 1, "dense_neurons": 64, "input_shape": (3, 224, 224)}),
    ]

    KFOLDS = 10
    EPOCHS = 100
    LR = 0.001 
    WEIGHT_DECAY = 0.0001
    BATCH_SIZE = 64
    PRE_RANDOM_SAMPLE = None
    DO_REBALANCE = True
    classes_to_drop = [13]

    mappings = load_mappings(pth=MAPPING_PATH)
    device = get_device()
    datasets = glob(str(PROCESSED_DATA_DIR / "all-classes_carson-224-3depths*.csv"))
    for model_name, model_class, architecture_params in MODELS:
        is_res_net = "ResNet" in model_name
        logger.info(f"MODEL: {model_name} | IS_RESNET: {is_res_net}")
        for dataset in datasets:
            files, labels = get_data_from_dataset_csv(dataset)

            embryo_names_to_files, embryo_names_to_count, embryo_names_to_labels = get_embryo_names_by_from_files(files, labels)
            logger.info(f"# EMBRYOS: {len(embryo_names_to_files)}")



            if PRE_RANDOM_SAMPLE:
                # from each class, randomly sample PRE_RANDOM_SAMPLE images
                # if a class has less than PRE_RANDOM_SAMPLE images, sample all, 
                # print out the number of images per class, and if the class has less than PRE_RANDOM_SAMPLE
                logger.info(f"PRE-SAMPLING: {PRE_RANDOM_SAMPLE}")
                files, labels = do_random_sample(PRE_RANDOM_SAMPLE, files, labels)

            dataset, num_classes = get_filename_no_ext(dataset), len(np.unique(labels))
            logger.info(f"DATASET {dataset} | NUM CLASSES: {num_classes}")

            mapping = mappings[dataset.split("_")[0]]
            class_names_by_label = get_class_names_by_label(mapping)
            logger.info(class_names_by_label)
            model_dir = configure_model_dir(model_name, dataset, mapping, architecture=architecture_params)
            logger.info(f"MODEL DIR: {model_dir}")
            
            # train model over k-folds, record performance
            accs, aucs, macros, losses = [], [], [], []
            conf_mats = np.zeros((num_classes, num_classes))

            #k_fold_splits = stratified_kfold_split(files, labels, n_splits=KFOLDS)
            k_fold_splits = []
            k_fold_splits_by_embryo = kfold_split(embryo_names_to_files.keys(), n_splits=KFOLDS)
            for train_embryos, val_embryos in k_fold_splits_by_embryo:
                train_files, train_labels = [], []
                for embryo in train_embryos:
                    if embryo_names_to_labels[embryo] not in classes_to_drop:
                        train_files.extend(embryo_names_to_files[embryo])
                        train_labels.extend(embryo_names_to_labels[embryo])
                val_files, val_labels = [], []
                for embryo in val_embryos:
                    if embryo_names_to_labels[embryo] not in classes_to_drop:
                        val_files.extend(embryo_names_to_files[embryo])
                        val_labels.extend(embryo_names_to_labels[embryo])
                k_fold_splits.append((train_files, train_labels, val_files, val_labels))


            for idx, fold in enumerate(k_fold_splits):
                # free up memory for new fold with pytorch
                torch.cuda.empty_cache()

                logger.info(f"Fold {idx+1}/{KFOLDS}")
                log_dir = f"{model_dir}/runs/fold_{idx}"
                logger.info(f"Tensorboard output >> {log_dir}")
                writer = SummaryWriter(log_dir=log_dir)
                train_ims, train_labels, val_ims, val_labels = fold

                if DO_REBALANCE:
                    balancer = DataBalancer(train_ims, train_labels, T=None, quartile=0.75, undersample=True, oversample=True, transforms=get_basic_transforms(),
                                            aug_dir= INTERIM_DATA_DIR / "aug")
                    balancer.print_before_and_after()
                    train_ims_new = balancer.balanced_img_paths()
                    train_labels_new = balancer.balanced_labels()
                else:
                    train_ims_new, train_labels_new = train_ims, train_labels
                
                train_data = CustomImageDataset(train_ims_new, train_labels_new, 
                                                img_transform=get_transforms(image_net_transforms=is_res_net), num_channels=3)

                val_data = CustomImageDataset(val_ims, val_labels, img_transform=get_transforms(image_net_transforms=is_res_net), num_channels=3)

                train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
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
                                                                    optimizer, device, criterion, False, 1, 1,
                                                                    EPOCHS, writer, best_model_path=model_save_path, 
                                                                    class_names=class_names_by_label, early_stop_epochs=20, 
                                                                    do_early_stop=True)
                # load best model and evaluate
                model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
                val_micro, val_aucs, val_macro, avg_loss, conf_mat = evaluate(model, device, val_loader, loss=criterion, get_conf_mat=True)
                val_auc=np.mean(val_aucs)
                
                logger.info(f'(Initial Performance Last Epoch) | test_loss={avg_loss:.4f} | test_micro={(val_micro * 100):.2f}, '
                                f'test_macro={(val_macro * 100):.2f}, test_auc={(val_auc * 100):.2f}')
                conf_mats += conf_mat
                accs.append(val_micro)
                aucs.append(val_auc)
                macros.append(val_macro)
                losses.append(avg_loss)
                writer.close()

                if DO_REBALANCE:
                    balancer.delete_augmentation()
                    del balancer
                
                del model, optimizer, criterion, train_loader, val_loader, train_data, val_data
            
            ### END of kFolds: Record model performance
            report_kfolds_results(model_dir, accs, aucs, macros, losses, conf_mats, KFOLDS)
            
            
            


