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
import nni
from embpred.config import MODELS_DIR
import torch.nn.functional as F

def configure_model_dir(model_name, dataset_name, mapping, additional_ids = None):
    """
    Configures a directory for the model based on its class name and dataset name.
    
    Parameters:
    - model_name: The name of the model being trained.
    - dataset_name: The name of the dataset being used for training.
    - mapping: A dictionary containing mappings that need to be saved.
    - additional_ids: a list of additional info that will form subdirectories

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
        raise ValueError(f"{new_model_dir} exists: {model_name} has already been trained with dataset {dataset_name}")
    os.makedirs(new_model_dir)
    mapping_file_path = os.path.join(new_model_dir, 'mapping.json')
    with open(mapping_file_path, 'w') as json_file:
        json.dump(mapping, json_file, indent=4)
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


def train_and_evaluate(model, train_loader, test_loader, optimizer, device, loss, use_nni, test_interval, epochs, writer: SummaryWriter):
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
    - test_interval: The interval (in epochs) at which to evaluate the model on the test set.
    - epochs: Total number of training epochs.
    - writer: SummaryWriter object for logging to TensorBoard.

    Returns:
    - Tuple containing mean accuracy, AUC, and macro F1-score over the test intervals.
    """
    model.train()
    accs, aucs, macros, losses = [], [], [], []
    for i in range(epochs):
        loss_all = 0
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            out = model(inputs)

            loss_value = loss(out, labels)
            loss_value.backward()
            optimizer.step()

            loss_all += loss_value.item()
        
        epoch_loss = loss_all / len(train_loader.dataset)
        writer.add_scalar('Loss/train', epoch_loss, i)

        train_micro, train_auc, train_macro, _ = evaluate(model, device, train_loader)
        logger.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}')
        
        writer.add_scalar('Metrics/Train_Micro_F1', train_micro, i)
        writer.add_scalar('Metrics/Train_Macro_F1', train_macro, i)
        writer.add_scalar('Metrics/Train_AUC', train_auc, i)

        if (i + 1) % test_interval == 0:
            test_micro, test_auc, test_macro, test_loss = evaluate(model, device, test_loader, loss=loss)
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

        if use_nni:
            nni.report_intermediate_result(train_auc)

    accs, aucs, macros, losses = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros)), np.sort(np.array(losses))
    return accs.mean(), aucs.mean(), macros.mean(), losses.mean()

import torch.nn.functional as F
from sklearn import metrics
import torch
import numpy as np

@torch.no_grad()
def evaluate(model, device, loader, loss=None, test_loader: Optional[torch.utils.data.DataLoader] = None) -> tuple:
    """
    Evaluates the model on the provided data loader.

    Parameters:
    - model: The PyTorch model to be evaluated.
    - device: The device on which to perform computations (CPU/GPU).
    - loader: DataLoader for the dataset to evaluate.
    - loss: criterion to calculate average loss function, if provided
    - test_loader: Optional DataLoader for an additional test set. If provided, both train and test metrics will be returned.

    Returns:
    - If test_loader is None, returns a tuple containing (train_micro, train_auc, train_macro, avg_loss).
    - If test_loader is provided, returns a tuple containing (train_micro, train_auc, train_macro, test_micro, test_auc, test_macro, avg_loss).
    """
    model.eval()
    preds, trues, preds_prob = [], [], []

    loss_all = 0
    total_samples = 0

    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        c = model(inputs)
        
        if loss is not None:
            loss_value = loss(c, labels)
            loss_all += loss_value.item() * inputs.size(0)  # Accumulate total loss weighted by batch size
            total_samples += inputs.size(0)

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
        epoch_loss = loss_all / total_samples  # Calculate the average loss over all samples
    else:
        epoch_loss = None

    # Calculate ROC AUC for multiclass with One-vs-Rest (OvR) strategy
    train_auc = metrics.roc_auc_score(trues, preds_prob, multi_class='ovr', average='weighted')

    # Calculate F1-scores
    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro')

    if test_loader is not None:
        test_micro, test_auc, test_macro, test_loss = evaluate(model, device, test_loader, loss)
        return train_micro, train_auc, train_macro, test_micro, test_auc, test_macro, epoch_loss
    else:
        return train_micro, train_auc, train_macro, epoch_loss
