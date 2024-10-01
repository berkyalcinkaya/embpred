import torch
import torch.nn as nn
import torchvision.models as models

ARCH_PARAMS = {"num_dense_layers": 1, "dense_neurons": 64} # model architecture parameters

def load_model(model_class, model_path, device, num_classes, model_params):
    """
    Load a model from a saved checkpoint.

    Parameters:
    - model_class: The class of the model to instantiate (use CustomResNet18, below).
    - model_path: The path to the checkpoint file.
    - device: The device to load the model onto (e.g., 'cpu' or 'cuda').
    - num_classes: The number of classes in the dataset.
    - model_params: The model architecture parameters (use the gloabl var ARCH_PARAMS).

    Returns:
    - model: The loaded model with the state dictionary applied.
    - epoch: The epoch at which the checkpoint was saved.
    - best_val_auc: The best validation AUC at the time the checkpoint was saved.
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Instantiate the model
    model = model_class(num_classes=10, **model_params).to(device)
    
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Retrieve additional information
    epoch = checkpoint['epoch']
    best_val_auc = checkpoint['best_val_auc']
    
    print(f"Model loaded from epoch {epoch} with best validation AUC: {best_val_auc:.4f}")
    
    return model, epoch, best_val_auc

class CustomResNet18(nn.Module):
    def __init__(self, num_classes, num_dense_layers, dense_neurons):
        super(CustomResNet18, self).__init__()
        # Load the pretrained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze all the ResNet-18 layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Remove the original fully connected layer
        num_ftrs = self.resnet.fc.in_features  # In ResNet-18, this is 512
        self.resnet.fc = nn.Identity()  # Replace the final fc layer with an identity layer

        # If a single integer is provided, apply it to all dense layers
        if isinstance(dense_neurons, int):
            dense_neurons = [dense_neurons] * num_dense_layers

        # Define the custom dense layers dynamically based on the specified number of layers
        layers = []
        input_size = num_ftrs
        for i, neurons in enumerate(dense_neurons):
            layers.append(nn.Linear(input_size, neurons))  # Fully connected layer
            layers.append(nn.ReLU(inplace=True))          # ReLU activation
            layers.append(nn.Dropout(0.5))                # Dropout
            input_size = neurons
        
        # Final output layer
        layers.append(nn.Linear(input_size, num_classes))  # Final output layer, no softmax
        
        # Store all layers in nn.Sequential
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the ResNet backbone
        x = self.resnet(x)
        
        # Forward pass through the custom classifier layers
        x = self.classifier(x)
        
        return x