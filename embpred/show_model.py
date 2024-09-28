import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from PIL import Image
from embpred.modeling.models import FirstNet
from embpred.features import extract_emb_frame_2d, load_faster_RCNN_model_device
import numpy as np
import cv2
import skimage

def load_model(model_class, model_path, device):
    """
    Load a model from a saved checkpoint.

    Parameters:
    - model_class: The class of the model to instantiate.
    - model_path: The path to the checkpoint file.
    - device: The device to load the model onto (e.g., 'cpu' or 'cuda').

    Returns:
    - model: The loaded model with the state dictionary applied.
    - epoch: The epoch at which the checkpoint was saved.
    - best_val_auc: The best validation AUC at the time the checkpoint was saved.
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Instantiate the model
    model = model_class(num_classes=10).to(device)
    
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Retrieve additional information
    epoch = checkpoint['epoch']
    best_val_auc = checkpoint['best_val_auc']
    
    print(f"Model loaded from epoch {epoch} with best validation AUC: {best_val_auc:.4f}")
    
    return model, epoch, best_val_auc

def load_data(csv_path):
    df = pd.read_csv(csv_path, header=None)  # No header, so we specify header=None
    df.sort_values(by=0, inplace=True)  # Sort by the first column (image paths)
    return df

def get_predictions(model, img, device, topk=3):
    # Resize to 800x800 as required
    img = cv2.resize(img, (800, 800))

    # Convert image to tensor
    transform = transforms.ToTensor()
    image = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()

    topk_probs, topk_indices = torch.topk(torch.tensor(probabilities), topk)
    return topk_probs.numpy(), topk_indices.numpy()

def display_image(img, label, predicted_classes=None, class_names=None):
    plt.figure(1, figsize=(20, 20))  # Reuse the same figure
    plt.clf()  # Clear the figure
    
    plt.imshow(img, cmap='gray')
    
    # Add the real label annotation
    plt.gca().text(
        0.05, 0.95, f"True Label: {class_names[label]}", 
        transform=plt.gca().transAxes, fontsize=12, color='white',
        bbox=dict(facecolor='green', alpha=0.5)
    )

    # If predictions are available, add them as annotations
    if predicted_classes is not None:
        for i, (prob, idx) in enumerate(predicted_classes):
            plt.gca().text(
                0.05, 0.90 - i * 0.05, f"{class_names[idx]}: {prob:.2f}" + (" (Highest)" if i == 0 else ""),
                transform=plt.gca().transAxes, fontsize=12, color='white',
                bbox=dict(facecolor='red' if i == 0 else 'blue', alpha=0.5)
            )
    plt.axis('off')
    plt.show(block=False)  # Show non-blocking for automatic updates
    plt.pause(0.001)  # Short pause to update the figure

def main(csv_path, model_path, class_names, device, n=20, delay=0.5):
    df = load_data(csv_path)
    model, _, _ = load_model(FirstNet, model_path, device)
    
    for i in range(0, len(df), n):
        img_path = df.iloc[i, 0]
        true_label = df.iloc[i, 1]
        
        # Load grayscale image as 2D array
        img = skimage.io.imread(img_path)
        print(img.shape)
        
        # Apply ExtractEmbFrame function to get the processed image
        rcnn_model, _ = load_faster_RCNN_model_device()
        processed_img = extract_emb_frame_2d(img, rcnn_model, device)
        
        print(processed_img.shape)
        
        # Show the image with the true label
        display_image(processed_img, true_label, class_names=class_names)
        
        time.sleep(delay)  # Delay before getting predictions
        
        # Get the model predictions
        probs, indices = get_predictions(model, processed_img, device)
        predicted_classes = list(zip(probs, indices))
        
        # Show the image with the predictions
        display_image(processed_img, true_label, predicted_classes=predicted_classes, class_names=class_names)
        
        time.sleep(2)  # Delay before moving to the next image
        
        # Clear the current figure (handled by display_image)

if __name__ == "__main__":
    # Example usage
    csv_path = "data/interim/sample_emb.csv"  # Path to the CSV file
    model_path = "models/FirstNet/detailed_classes_tEmpty_t2_t3_t4_t5_t6_t7_t8_tPN_tPNF/sampled/runs/fold_0/model.pth"  # Path to the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = {
        0: "t1",
        1: "t2",
        2: "t3",
        3: "t4",
        4: "t5",
        5: "t6",
        6: "t7", 
        7: "t8",
        8: "tPN",
        9: "tPNf"
    }

    main(csv_path, model_path, class_names, device, n=5, delay=7)