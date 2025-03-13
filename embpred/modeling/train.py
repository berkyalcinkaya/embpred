from gc import freeze
from itertools import count
from platform import architecture
import random
from loguru import logger
from requests import get
import test
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
from embpred.modeling.train_utils import get_device, train_and_evaluate, evaluate, configure_model_dir, write_data_split
from embpred.modeling.utils import report_kfolds_results, report_test_set_results
from embpred.modeling.loss import get_class_weights, weighted_cross_entropy_loss
import csv

# TODO: add a test set for when k_folds > 2


# write code to ensure that pytorch caches are cleared and that the GPU memory is freed up
torch.cuda.empty_cache()

MAPPING_PATH = RAW_DATA_DIR / "mappings.json"

VAL_SIZE = 0.1
TEST_SIZE = 0.15
KFOLDS = 1
EPOCHS = 20
LR = 0.0001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
PRE_RANDOM_SAMPLE = None
DO_REBALANCE = True
DEBUG = False
if DEBUG:
    EPOCHS = 1
EARLY_STOP_EPOCHS = 10
REMOVE_DIR = True


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
        #("ResNet50Unfreeze-CE-embSplits-balanced-1layer64", CustomResNet50, {"num_dense_layers": 1, "dense_neurons": 64, "freeze_": False}),
        ("ResNet50Unfreeze-weightCE-embSplits-balanced-0layer", CustomResNet50, {"num_dense_layers": 0, "dense_neurons": True, "freeze_": False})
       #("ResNet50-CE-embSplits-fullbalanced-1layer64", CustomResNet50, {"num_dense_layers": 1, "dense_neurons": 64})
        #("ResNet50-CE-embSplits-fullbalanced-1layer128", CustomResNet50, {"num_dense_layers": 1, "dense_neurons": 128}),
        #("ResNet50-CE-embSplits-fullbalanced-2layer256-128", CustomResNet50, {"num_dense_layers": 2, "dense_neurons": [256,128]}),
        #("ResNet50-CE-embSplits-fullbalanced-2layer128-64", CustomResNet50, {"num_dense_layers": 2, "dense_neurons": [128,64]}),
        #("ResNet50-CE-embSplits-fullbalanced-3layer512-256-128", CustomResNet50, {"num_dense_layers": 3, "dense_neurons": [512,256,128]}),
        # add drop out to all the models above
        #("ResNet50-CE-embSplits-fullbalanced-1layer64-dropout50", CustomResNet50, {"num_dense_layers": 1, "dense_neurons": 64, "dropout":True, "dropout_rate":0.5}),
        #("ResNet50-CE-embSplits-fullbalanced-1layer128-dropout50", CustomResNet50, {"num_dense_layers": 1, "dense_neurons": 128, "dropout":True, "dropout_rate":0.5}),
        #("ResNet50-CE-embSplits-fullbalanced-2layer256-128-dropout50", CustomResNet50, {"num_dense_layers": 2, "dense_neurons": [256,128], "dropout_rate":0.5})
        #("ResNet50-CE-embSplits-fullbalanced-2layer128-64-dropout50", CustomResNet50, {"num_dense_layers": 2, "dense_neurons": [128,64], "dropout":True, "dropout_rate":0.5})

        #("SimpleNet3D-weightLoss-embSplits", SimpleNet3D, {})
        #("Wnet-weight-noXaiver-embSplits", WNet, {"dropout":False, "dropout_rate":0.5, "do_xavier":False})
        #("BigWnet-embSplits", BigWNet, {"dropout":False})
        #("ResNet50-weightLoss-2layer-embSplits", CustomResNet50, {"num_dense_layers": 2, "dense_neurons": [128,64]}),
        #("ResNet50-weightLoss-1layer-embSplits", CustomResNet50, {"num_dense_layers": 1, "dense_neurons": 64}),
        #("Wnet-drop-weightLoss-drop-embSplits", WNet, {"dropout":True, "dropout_rate":0.5})
        #("BiggerNet3D224-emb-kfolds-noUpSample", BiggerNet3D224, {})
        #("CustomResNet18-1layer-full-balance", CustomResNet18, {"num_dense_layers": 1, "dense_neurons": 64, "input_shape": (3, 224, 224)}),
    ]

    mappings = load_mappings(pth=MAPPING_PATH)
    device = get_device()
    datasets = [PROCESSED_DATA_DIR / "all-classes_carson-224-3depths.csv"] #, PROCESSED_DATA_DIR / "all-classes_carson-224-3depths-noCrop.csv"] # dataset prefixes determines which mapping to use
    for dataset in datasets:
        assert(os.path.exists(dataset))
    
    for model_name, model_class, architecture_params in MODELS:
        criterion = weighted_cross_entropy_loss(14, [1,2,4,6,7,8,9,11,12], device, weight_noisy=0.5, weight_clean=1)  # nn.CrossEntropyLoss() # #nn.CrossEntropyLoss()#(14, classes_to_drop, weight_clean=1.0, weight_noisy=0.5)
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
            model_dir = configure_model_dir(model_name, dataset, mapping, architecture=architecture_params, 
                                            debug=DEBUG, remove=REMOVE_DIR)
            logger.info(f"MODEL DIR: {model_dir}")
            
            # train model over k-folds, record performance
            accs, aucs, macros, losses = [], [], [], []
            conf_mats = np.zeros((num_classes, num_classes))

            #k_fold_splits = stratified_kfold_split(files, labels, n_splits=KFOLDS)
            embryos = list(embryo_names_to_files.keys())
            if DEBUG:
                logger.info("DEBUG MODE: ONLY TRAINING ON 10 EMBRYOS")
                embryos = embryos[:50]
            k_fold_splits = []
            k_fold_splits_by_embryo = kfold_split(embryos, n_splits=KFOLDS, random_state=RANDOM_STATE, val_size=VAL_SIZE, test_size=TEST_SIZE)
            for train_embryos, val_embryos, test_embryos in k_fold_splits_by_embryo:
                
                # TODO: this is hacked for the n_splits = 1 case
                logger.info(f"Train: {len(train_embryos)} | Val: {len(val_embryos)} | Test: {len(test_embryos)}")
                write_data_split( train_embryos, val_embryos, test_embryos, model_dir)
                
                train_files, train_labels = [], []
                for embryo in train_embryos:
                    train_files.extend(embryo_names_to_files[embryo])
                    train_labels.extend(embryo_names_to_labels[embryo])
                val_files, val_labels = [], []
                for embryo in val_embryos:
                    val_files.extend(embryo_names_to_files[embryo])
                    val_labels.extend(embryo_names_to_labels[embryo])
                test_files, test_labels = [], []
                for embryo in test_embryos:
                    test_files.extend(embryo_names_to_files[embryo])
                    test_labels.extend(embryo_names_to_labels[embryo])
                k_fold_splits.append((train_files, train_labels, val_files, val_labels, test_files, test_labels))
            
            for idx, fold in enumerate(k_fold_splits):
                # free up memory for new fold with pytorch
                torch.cuda.empty_cache()

                logger.info(f"Fold {idx+1}/{KFOLDS}")
                log_dir = f"{model_dir}/runs/fold_{idx}"
                logger.info(f"Tensorboard output >> {log_dir}")
                writer = SummaryWriter(log_dir=log_dir)
                train_ims, train_labels, val_ims, val_labels, test_ims, test_labels = fold
                has_test = len(test_ims) > 0
                logger.info(f"Train: {len(train_ims)} | Val: {len(val_ims)} | Test: {len(test_ims)}")

                if DO_REBALANCE:
                    balancer = DataBalancer(train_ims, train_labels, T=2500, quartile=0.75, undersample=True, oversample=True, transforms=get_basic_transforms(),
                                            aug_dir= INTERIM_DATA_DIR / "aug")
                    balancer.print_before_and_after()
                    train_ims_new = balancer.balanced_img_paths()
                    train_labels_new = balancer.balanced_labels()
                else:
                    train_ims_new, train_labels_new = train_ims, train_labels
                
                train_data = CustomImageDataset(train_ims_new, train_labels_new, num_classes,
                                                img_transform=get_transforms(image_net_transforms=is_res_net), num_channels=3)
                val_data = CustomImageDataset(val_ims, val_labels, num_classes, img_transform=get_transforms(image_net_transforms=is_res_net), num_channels=3)
                test_data = CustomImageDataset(test_ims, test_labels, num_classes, img_transform=get_transforms(image_net_transforms=is_res_net), num_channels=3)


                train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=4)
                test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4)

                model = model_class(num_classes=num_classes, **architecture_params)
                param_count = count_parameters(model)
                logger.info(f"{str(model)}")
                logger.info(f"Model parameters: {param_count}")
                model.to(device)
                
                logger.info(f"Loaded model to {device}")
                
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                
                model_save_path = os.path.join(log_dir, "model.pth")
                logger.info(f"BEST WEIGHTS: {model_save_path}")
        
                val_micro, val_auc, val_macro, val_losses = train_and_evaluate(model, train_loader, val_loader, 
                                                                    optimizer, device, criterion, False, 1, 1,
                                                                    EPOCHS, writer, best_model_path=model_save_path, 
                                                                    class_names=class_names_by_label, early_stop_epochs=EARLY_STOP_EPOCHS, 
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
                
                del model, optimizer, train_loader, val_loader, train_data, val_data
            
            ### END of kFolds: Record model performance
            logger.info("KFOLDS COMPLETE: REPORTING RESULTS...")
            model = model_class(num_classes=num_classes, **architecture_params)
            model.to(device)
            model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
            test_micro, test_aucs, test_macro, avg_loss, conf_mat = evaluate(model, device, test_loader, loss=criterion, get_conf_mat=True)
            test_auc = np.mean(test_aucs)
            report_kfolds_results(model_dir, accs, aucs, macros, losses, conf_mats, KFOLDS)
            report_test_set_results(model_dir, test_micro, test_auc, test_macro, avg_loss, conf_mat)
            del test_micro, test_aucs, test_macro, avg_loss, conf_mat
            del test_loader, test_data, model
            torch.cuda.empty_cache()
        

            
            


