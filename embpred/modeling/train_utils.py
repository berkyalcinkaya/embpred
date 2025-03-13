from calendar import c
from loguru import logger
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from sklearn import metrics
from torch.utils.data import DataLoader
import json 
import os
#import nni
from embpred.config import MODELS_DIR, DATA_DIR
import torch.nn.functional as F
from tqdm import tqdm
import shutil


def write_data_split(train_embryos, val_embryos, test_embryos, loc=DATA_DIR):
    with open(loc / "train_val_test_embryos.txt", "w") as f:
        f.write("train\n")
        for embryo in train_embryos:
            f.write(embryo + "\n")
        f.write("val\n")
        for embryo in val_embryos:
            f.write(embryo + "\n")
        f.write("test\n")
        for embryo in test_embryos:
            f.write(embryo + "\n")
    logger.info("Wrote train, val, and test embryo names to {}".format(DATA_DIR / "train_val_test_embryos.txt"))



def save_best_model(state, filename):
    torch.save(state, filename)
    
def write_aucs_by_class(aucs, epoch, writer, mode="Train", mappings=None):
    for i, auc in enumerate(aucs):
        if mappings:
            class_name = mappings[i]
        else:
            class_name = f"class{i}"
        writer.add_scalar(f"Metrics/{mode}_AUC_{class_name}", auc, epoch)
        
def configure_model_dir(model_name, dataset_name, mapping, architecture=None, additional_ids = None, debug=False):
    """
    Configures a directory for the model based on its class name and dataset name.
    
    Parameters:
    - model_name: The name of the model being trained.
    - dataset_name: The name of the dataset being used for training.
    - mapping: A dictionary containing mappings that need to be saved.
    - architecture: a dictionary containing the architecture of the model
    - additional_ids: a list of additional info that will form subdirectories
    - debug: a boolean flag that specifies whether the model is being trained in debug mode, if so
            model_dir is deleted if it already exists

    Returns:
    - new_model_dir: The path to the newly created model directory.

    Raises:
    - ValueError: If the directory for the given model class and dataset already exists.
    """
    new_model_dir = os.path.join(MODELS_DIR, model_name, dataset_name)

    if additional_ids:
        for id in additional_ids:
            new_model_dir = os.path.join(new_model_dir, id)

    if os.path.exists(new_model_dir):
        if debug:
            logger.info(f"Deleting {new_model_dir}")
            shutil.rmtree(new_model_dir)
            os.makedirs(new_model_dir)
        else:
            raise ValueError(f"{new_model_dir} exists: {model_name} has already been trained with dataset {dataset_name}")
    else:
        os.makedirs(new_model_dir)
    mapping_file_path = os.path.join(new_model_dir, 'mapping.json')
    with open(mapping_file_path, 'w') as json_file:
        json.dump(mapping, json_file, indent=4)
    
    # if architecture exists, save it to a json file
    if architecture:
        arch_file_path = os.path.join(new_model_dir, 'architecture.json')
        with open(arch_file_path, 'w') as json_file:
            json.dump(architecture, json_file, indent=4)

    return new_model_dir

def get_device():
    """
    Returns the appropriate device ('cuda' if available, otherwise 'cpu') for training.

    Returns:
    - device: The torch device to be used for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU {device} found")
    else:
        device = torch.device("cpu")
        logger.info("Using with CPU")
    return device


def train_and_evaluate(model, train_loader, test_loader, optimizer, device, loss, use_nni, train_test_interval, test_interval, 
                       epochs, writer: SummaryWriter, best_model_path=None, class_names=None, do_early_stop=False,
                       early_stop_epochs=None):
    """
    Trains the model and evaluates its performance at specified intervals.

    Parameters:
    - model: The PyTorch model to be trained.
    - train_loader: DataLoader for the training dataset.
    - test_loader: DataLoader for the test dataset.
    - optimizer: The optimizer used for training.
    - device: The device on which to perform computations (CPU/GPU).
    - loss: The loss function to be used.
    - use_nni: Boolean indicating whether to report intermediate results to NNI.
    - train_test_interval: The interval (in epochs) at which to evaluate the model on the training set
    - test_interval: The interval (in epochs) at which to evaluate the model on the test set.
    - epochs: Total number of training epochs.
    - writer: SummaryWriter object for logging to TensorBoard.
    - best_model_path: if specified, save model checkpoint when new auc best is acheived
    - class_names: a dict or list of strings. A dictionary should specify the numeric label as
                            the key and name as the value. A list should be of length num_classes and
                            contain the class name by index. Optional, if provided, used to write aucs
                            to tensorboard
    - do_early_stop (bool, default False): whether to implement an early stopping callback. If true, user should specify early_stop_epochs
    - early_stop_epochs (Optional int): maximum number of epochs since model improvement in terms of test_auc before training is cut short. 
                                        Only used if do_early_stop is set to True

    Returns:
    - Tuple containing mean accuracy, AUC, macro F1-score, and loss over the test intervals.
    """
    model.train()
    accs, aucs, macros, losses = [], [], [], []
    
    logger.info("Starting training")
    
    best_auc = 0
    num_epochs_since_improvement = 0
    for i in range(epochs):
        
        if do_early_stop and num_epochs_since_improvement >= early_stop_epochs:
            logger.info(f"Early stopping at epoch {i}")
            break
        
        loss_all = 0
        for data in tqdm(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            out = model(inputs)

            loss_value = loss(out, labels)
            loss_value.backward()
            optimizer.step()

            loss_all += loss_value.item()
        
        epoch_loss = loss_all / len(train_loader.dataset)
        
        # log the training loss
        # AUCs, F1-scores, and confusion matrices are calculated at the specified test intervals
        writer.add_scalar('Loss/train', epoch_loss, i)
        train_loss_str = f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}'
        if (i + 1) % train_test_interval == 0:
            train_micro, train_aucs, train_macro, _, _ = evaluate(model, device, train_loader, get_conf_mat=False)
            train_auc = np.mean(train_aucs)
            logger.info(f'{train_loss_str}, '
                        f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                        f'train_auc={(train_auc * 100):.2f}')
            
            writer.add_scalar('Metrics/Train_Micro_F1', train_micro, i)
            writer.add_scalar('Metrics/Train_Macro_F1', train_macro, i)
            writer.add_scalar('Metrics/Train_AUC', train_auc, i)
            write_aucs_by_class(train_aucs, i, writer, mode="Train", mappings=class_names)
        else:
            logger.info(train_loss_str)

        # Evaluate the model on the test set if the test interval is reached
        if (i + 1) % test_interval == 0:
            test_micro, test_aucs, test_macro, test_loss, _ = evaluate(model, device, test_loader, loss=loss, get_conf_mat=False)
            
            test_auc = np.mean(test_aucs)
            if test_auc > best_auc:
                best_auc = test_auc
                num_epochs_since_improvement = 0
                
                if best_model_path is not None:
                    save_best_model({
                                    'epoch': i,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'best_val_auc': best_auc,
                                }, best_model_path)
            else:
                num_epochs_since_improvement +=test_interval
                
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
            losses.append(test_loss)
            logger.info(f'(Train Epoch {i}), test_loss={test_loss:.4f}, test_micro={(test_micro * 100):.2f}, ' \
                        f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}')
            writer.add_scalar('Loss/test', test_loss, i)
            writer.add_scalar('Metrics/Test_Micro_F1', test_micro, i)
            writer.add_scalar('Metrics/Test_Macro_F1', test_macro, i)
            writer.add_scalar('Metrics/Test_AUC', test_auc, i)
            write_aucs_by_class(test_aucs, i, writer, mode="Test", mappings=class_names)

        if use_nni:
            continue
            #nni.report_intermediate_result(train_auc)

    accs, aucs, macros, losses = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros)), np.sort(np.array(losses))
    return accs.mean(), aucs.mean(), macros.mean(), losses.mean()


@torch.no_grad()
def evaluate(model, device, loader, loss=None, test_loader: Optional[torch.utils.data.DataLoader] = None, get_conf_mat:Optional[bool]=True) -> tuple:
    """
    Evaluates the model on the provided data loader.

    Parameters:
    - model: The PyTorch model to be evaluated.
    - device: The device on which to perform computations (CPU/GPU).
    - loader: DataLoader for the dataset to evaluate.
    - loss: criterion to calculate average loss function, if provided
    - test_loader: Optional DataLoader for an additional test set. If provided, both train and test metrics will be returned.
    - get_conf_mat: boolean flag that specifies whether or not a confusion matrix should be return. Default, True (confusion matrix returned)

    Returns:
    - If test_loader is None, returns a tuple containing (train_micro, train_aucs, train_macro, avg_loss, conf_mat).
    - If test_loader is provided, returns a tuple containing (train_micro, train_aucs, train_macro, test_micro, test_auc, test_macro, avg_loss, conf_mat).
    - If get_conf_mat is True, the tuple will contain a final entry of an np.ndarray of shape (num_classes x num_classes), contianing the confusion matrix. 
        Otherwise, conf_mat is None
    """
    model.eval()
    preds, trues, preds_prob = [], [], []
    conf_mat = None

    loss_all = 0

    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        c = model(inputs)
        
        if loss is not None:
            loss_value = loss(c, labels)
            loss_all += loss_value.item()

        # Apply softmax to get probabilities for each class
        probabilities = F.softmax(c, dim=1).detach().cpu().numpy()

        # Get the predicted classes
        pred = probabilities.argmax(axis=1)
        preds.extend(pred)

        # Convert one-hot encoded labels to class indices
        labels = torch.argmax(labels, dim=1).detach().cpu().numpy()
        trues.extend(labels)

        # Store the probabilities for each class (required for ROC AUC)
        preds_prob.extend(probabilities)

    # Convert lists to numpy arrays
    preds_prob = np.array(preds_prob)
    trues = np.array(trues)
    preds = np.array(preds)

    

    if loss is not None:
        epoch_loss = loss_all / len(loader.dataset)# Calculate the average loss over all samples
    else:
        epoch_loss = None
    
    if get_conf_mat:
        logger.info("getting conf mat")
        conf_mat = metrics.confusion_matrix(trues, preds)

    # Calculate ROC AUC for multiclass with One-vs-Rest (OvR) strategy
    train_aucs = metrics.roc_auc_score(trues, preds_prob, multi_class='ovr', average=None)

    # Calculate F1-scores
    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro')

    if test_loader is not None:
        test_micro, test_auc, test_macro, test_loss, _ = evaluate(model, device, test_loader, loss)
        return train_micro, train_aucs, train_macro, test_micro, test_auc, test_macro, epoch_loss, conf_mat
    else:
        return train_micro, train_aucs, train_macro, epoch_loss, conf_mat
