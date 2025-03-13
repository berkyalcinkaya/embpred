import torch
import torch.nn as nn

def get_class_weights(num_classes: int, noisy_classes: list, device: torch.device, weight_clean: float = 1.0, weight_noisy: float = 0.5, weight_dict=None) -> torch.FloatTensor:
    """
    Computes class weights for a weighted cross entropy loss.
    
    Each class is assigned a weight. For classes specified in noisy_classes,
    a lower weight is assigned (default weight_noisy=0.5) and for the other classes,
    a clean weight is assigned (default weight_clean=1.0).
    
    Parameters:
        num_classes (int): Total number of classes. Classes are assumed to be numbered from 0 to num_classes-1.
        noisy_classes (list): List of class indices considered noisy.
        device (torch.device): Device to move the weights tensor to.
        weight_clean (float, optional): Weight to assign to non-noisy classes. Default is 1.0.
        weight_noisy (float, optional): Weight to assign to noisy classes. Default is 0.5.
        weight_dict (dict, optional): Dictionary to assign custom weights to specific classes.
    
    Returns:
        torch.FloatTensor: A tensor of shape (num_classes,) containing the weights for each class.
    """
    weights = []
    for i in range(num_classes):
        if weight_dict is not None and i in weight_dict:
            weights.append(weight_dict[i])
        elif i in noisy_classes:
            weights.append(weight_noisy)
        else:
            weights.append(weight_clean)
    return torch.FloatTensor(weights).to(device)

def weighted_cross_entropy_loss(num_classes: int, noisy_classes: list, device: torch.device,  weight_clean: float = 1.0,
                                weight_noisy: float = 0.5, weight_dict = None) -> nn.CrossEntropyLoss:
    """
    Creates a weighted CrossEntropyLoss using class weights computed based on noisy classes.
    
    Parameters:
        num_classes (int): Total number of classes.
        noisy_classes (list): List of class indices that are considered noisy.
        device (torch.device): Device to move the weights tensor to.
        weight_clean (float, optional): Weight to assign to non-noisy classes. Default is 1.0.
        weight_noisy (float, optional): Weight to assign to noisy classes. Default is 0.5.
        weight_dict (dict, optional): Dictionary to assign custom weights to specific classes.
    
    Returns:
        nn.CrossEntropyLoss: A PyTorch CrossEntropyLoss object that uses the computed class weights.
    """
    class_weights = get_class_weights(num_classes, noisy_classes, device, weight_clean, weight_noisy, weight_dict=weight_dict)
    # Ensure you move the weights tensor to the same device as your model, e.g., using .to(device) when training.
    return nn.CrossEntropyLoss(weight=class_weights)