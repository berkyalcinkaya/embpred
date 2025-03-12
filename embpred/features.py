from pathlib import Path
import numpy as np
import typer
from loguru import logger
import os
from tqdm import tqdm
import cv2
from glob import glob
from embpred.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RCNN_PATH
import torch
from torchvision import transforms
from skimage.io import imread
from torchvision import models, transforms

def get_temporal_embedding()

def create_resnet50_embedding(image: np.ndarray) -> torch.Tensor:
    """
    Creates an image embedding from a 224x224x3 image using a pre-trained ResNet50.
    The function accepts an image as a NumPy array, converts it to a tensor,
    scales it, and applies normalization before obtaining the embedding.

    Parameters:
    - image (np.ndarray): Input image of shape (224, 224, 3).

    Returns:
    - embedding (torch.Tensor): The embedding vector produced by ResNet50.
    """
    # Convert NumPy array with shape (H, W, C) to a torch.Tensor with shape (C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    # Scale pixel values from [0, 255] to [0, 1]
    image_tensor = image_tensor / 255.0

    # Define normalization transform for ResNet50
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_tensor = normalize(image_tensor)
    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Load pre-trained ResNet50 model and remove the final fully connected layer
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # Replace classification head with identity to get embeddings
    model.eval()  # Set model to evaluation mode

    # Obtain the embedding without computing gradients
    with torch.no_grad():
        embedding = model(image_tensor)
    
    # Remove the batch dimension before returning the embedding
    return embedding.squeeze(0)


def load_faster_RCNN_model_device(use_GPU=True):
    if use_GPU:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
        
    model = torch.load(RCNN_PATH, map_location=device)
    return model,device

def extract_emb_frame_2d(embframe, model, device):# what is return type of this function? 

    return ExtractEmbFrame(embframe, embframe, embframe, model, device)[0]

def ExtractEmbFrame(r_channel, g_channel, b_channel, model, device):
    
    r_rgb = cv2.cvtColor(r_channel, cv2.COLOR_GRAY2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_tensor = transform(r_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    best_bbox = None
    best_score = 0
    for bbox, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
        if score > best_score:
            best_bbox = bbox
            best_score = score

    if best_bbox is None:
        
        padded_r = np.zeros((800, 800), dtype=np.uint8) # update the size
        padded_g = padded_r
        padded_b = padded_r
        
        return padded_r, padded_g, padded_b

    else:
        
        best_bbox = best_bbox.cpu().numpy()

        x_min, y_min, x_max, y_max = best_bbox.astype(int)
        cropped_r = r_channel[y_min:y_max, x_min:x_max]
        cropped_g = g_channel[y_min:y_max, x_min:x_max]
        cropped_b = b_channel[y_min:y_max, x_min:x_max]

        h, w = cropped_r.shape
    
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            pad_top = pad_bottom = 0
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            pad_left = pad_right = 0
    
        padded_r = cv2.copyMakeBorder(
            cropped_r,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )
        padded_g = cv2.copyMakeBorder(
            cropped_g,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )
        padded_b = cv2.copyMakeBorder(
            cropped_b,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )

        return padded_r, padded_g, padded_b

def check_img_features(dir = INTERIM_DATA_DIR / "EmbStages1_Focused"):
    im_shape = None
    num_incorrect = 0
    records = {}
    im_files = glob(os.path.join(dir, "*.jpeg"))
    shapes = []
    incorrect_im_files = []
    for im_file in im_files:
        im = imread(im_file)
        if im_shape is None:
            im_shape = im.shape

        shapes.append(im.shape)
        
        if im_shape != im.shape:
            num_incorrect+=1

            if im.shape in records:
                records[im.shape].append(im_file)
            else:
                records[im.shape] = [im_file]
    print(f"{num_incorrect}/{len(im_files)}")
    return records, shapes

##############################################################################################################################