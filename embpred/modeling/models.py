import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class FirstNet2D(nn.Module):

    def __init__(self, num_classes=4):
        super(FirstNet2D, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Update the in_features based on the input size and conv layers
        self.fc1 = nn.Linear(16 * 197 * 197, 120)  # Updated based on the new image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 796, 796), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 398, 398) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 394, 394) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 197, 197) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 16*197*197) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 16*197*197) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Output layer: (N, 84) Tensor input, and
        # outputs a (N, num_classes) Tensor
        output = self.fc3(f6)
        return output

class BiggerNet3D(nn.Module):
    def __init__(self, num_classes=10):  # You can specify the number of classes here
        super(BiggerNet3D, self).__init__()
        # Define the first convolutional layer: 3 input channels, 8 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 8, 5)
        # Define the second convolutional layer: 8 input channels, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(8, 32, 5)
        # Define the third convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, 3)
        # Define a max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with updated sizes based on input image size (3x800x800)
        self.fc1 = nn.Linear(64 * 97 * 97, 256)  # Update input features to match the output from the conv layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply the first convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply the third convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Apply fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (no activation function here because we'll use CrossEntropyLoss, which includes softmax)
        x = self.fc3(x)
        return x


class BiggerNet3D224(nn.Module):
    def __init__(self, num_classes=10):  # You can specify the number of classes here
        super(BiggerNet3D224, self).__init__()
        # Define the first convolutional layer: 3 input channels, 8 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 8, 5)
        # Define the second convolutional layer: 8 input channels, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(8, 32, 5)
        # Define the third convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, 3)
        # Define a max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with updated sizes based on input image size (3x224x224)
        # After three convolutional layers and pooling, the image size will be reduced to 26x26
        self.fc1 = nn.Linear(64 * 26 * 26, 256)  # Updated to 64 * 26 * 26 to match new conv output
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        print(x.shape)
        # Apply the first convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply the third convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Apply fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (no activation function here because we'll use CrossEntropyLoss, which includes softmax)
        x = self.fc3(x)
        return x



class SimpleNet3D(nn.Module):
    def __init__(self, num_classes=10):  # You can specify the number of classes here
        super(SimpleNet3D, self).__init__()
        # Define the first convolutional layer: 3 input channels, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Define the second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Define a max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with updated sizes based on input image size (3x800x800)
        self.fc1 = nn.Linear(16 * 197 * 197, 120)  # Input size is calculated from conv layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Apply the first convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Apply fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (no activation function here because we'll use CrossEntropyLoss, which includes softmax)
        x = self.fc3(x)
        return x
    

import torch
import torch.nn as nn
import torchvision.models as models

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



class CustomResNet50(nn.Module):
    def __init__(self, num_classes, num_dense_layers, dense_neurons):
        super(CustomResNet50, self).__init__()
        # Load the pretrained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True) #, num_classes=num_classes)
        
        # Freeze all the ResNet-50 layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Remove the original fully connected layer
        num_ftrs = self.resnet.fc.in_features  # In ResNet-50, this is 2048
        self.resnet.fc = nn.Identity()  # Replace the final fc layer with an identity layer

        # If a single integer is provided, apply it to all dense layers
        if isinstance(dense_neurons, int):
            dense_neurons = [dense_neurons] * num_dense_layers

        # Define the custom dense layers dynamically based on the specified number of layers
        layers = []
        input_size = num_ftrs
        for i, neurons in enumerate(dense_neurons):
            layers.append(('fc{}'.format(i + 1), nn.Linear(input_size, neurons)))
            layers.append(('relu{}'.format(i + 1), nn.ReLU(inplace=True)))
            layers.append(('dropout{}'.format(i + 1), nn.Dropout(0.5)))
            input_size = neurons
        layers.append(('fc_final', nn.Linear(input_size, num_classes)))  # Final output layer, no softmax
        
        self.classifier = nn.Sequential(*[nn.Sequential(layer) for layer in layers])

    def forward(self, x):
        # Forward pass through ResNet-50 backbone (without the final FC layer)
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # Flatten to a vector
        
        # Forward pass through custom classifier
        x = self.classifier(x)
        return x  # Return logits, no softmax