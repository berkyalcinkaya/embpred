from stringprep import map_table_b2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

RCNN_PATH = 'Faster_RCNN.pt'

def load_faster_RCNN_model_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(RCNN_PATH, map_location=device)
    return model,device

def extract_emb_frame_2d(embframe, model, device):
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
