from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from glob import glob
import os
from typing import List, Union
from embpred.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR, INTERIM_DATA_DIR
from embpred.features import ExtractEmbFrame, extract_emb_frame_2d, load_faster_RCNN_model_device
import csv
import json
import torch
import cv2 
from skimage.io import imsave, imread
import numpy as np
import shutil
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.utils import resample

def focus_and_pad(image, target_size, model, device):
    for i in [0,1]:
        assert(image.shape[i] <= target_size[i])
    image_focused = extract_emb_frame_2d(image, model, device)
    return get_padding(image_focused, target_size)

def get_padding(image, target_size):
    # Assuming image is a NumPy array with shape (height, width, channels) or (height, width)
    image_height, image_width = image.shape[:2]

    # Calculate padding values
    pad_left = int((target_size[0] - image_width) / 2)
    pad_top = int((target_size[1] - image_height) / 2)
    pad_right = int((target_size[0] - image_width) / 2 + (target_size[0] - image_width) % 2)
    pad_bottom = int((target_size[1] - image_height) / 2 + (target_size[1] - image_height) % 2)

    # Pad the image using np.pad
    if len(image.shape) == 3:  # If the image has channels (e.g., RGB)
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:  # If the image is grayscale
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    padded_image = np.pad(image, padding, mode='constant', constant_values=0)  # You can change the padding mode and value

    return padded_image


def pad_images_in_directory(input_dir, output_dir=None, target_size=(800, 800), replace_original=False):
    """
    Pads all JPEG images in the specified input directory to the target size and saves them to the output directory.
    Optionally, replaces the original images with the padded versions.

    Parameters:
    - input_dir: Directory containing the input JPEG images.
    - output_dir: Directory where the padded images will be saved. Ignored if replace_original is True.
    - target_size: Tuple specifying the target size (width, height) to pad the images to. Default is (800, 800).
    - replace_original: If True, the original images will be replaced with the padded images. Default is False.
    """
    # If replacing original images, set output_dir to input_dir
    if replace_original:
        output_dir = input_dir
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    # Define the padding transform

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            file_path = os.path.join(input_dir, filename)
            image = Image.open(file_path)
            
            # Apply padding
            padding = get_padding(image, target_size)
            padded_image = padding(image)
            
            # Save the padded image to the output directory
            output_path = os.path.join(output_dir, filename)
            padded_image.save(output_path)
            print(f"Saved padded image to: {output_path}")


def crop_bottom_pixels(image, num_pix = 25):
    # Crop the bottom 25 pixels
    return image[:(-1*num_pix), :]

def extract_embryos(paths, labels, dataset_name = "EmbStages1_Focused"):
    new_paths = []
    new_dir = INTERIM_DATA_DIR / dataset_name

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(MODELS_DIR / 'FasterRCNN.pt')
    for path, label in tqdm(zip(paths, labels)):
        Img = cv2.imread(path)
        emb_frame, _, _ = ExtractEmbFrame(Img[:,:,0] ,Img[:,:,1], Img[:,:,2], model , device)
        _, name = os.path.split(path)
        new_name = new_dir / f"{label}_{name}"
        #print(emb_frame.shape, new_name)
        imsave(new_name, emb_frame)
        new_paths.append(path)


def get_all_image_paths_from_raws(loc = RAW_DATA_DIR / "EmbStages/dataset1", im_type = "jpeg"):
    stage_dirs = glob(str(loc / "*/"))
    stage_names = [p for p in os.listdir(loc)]
    labels = []
    curr_label = 0
    paths = []

    for stage_dir, stage_name in tqdm(zip(sorted(stage_dirs), sorted(stage_names))):
        im_files = glob(os.path.join(stage_dir, f"*.{im_type}"))
        paths += im_files
        labels += [stage_name for _ in range(len(im_files))]
        curr_label += 1
    return paths, labels

def get_all_images_interim(dataset = "EmbStages1_Focused", ext = "jpeg"):
    img_dir = INTERIM_DATA_DIR  / dataset
    img_tags = img_dir / f"*{ext}"
    all_files = sorted(glob(str(img_tags)))
    paths = []
    labels = []
    for file in tqdm(all_files):
        im_class = os.path.split(file)[-1].split("_")[0]
        paths.append(file)
        labels.append(im_class)
    return paths, labels


def labels_to_numeric(labels, dataset_mapping):
    return [dataset_mapping[label] for label in labels]

def make_dataset(mappings: List[dict], paths, labels, loc = PROCESSED_DATA_DIR, dataset_additional_text=None):
    for mapping_name in tqdm(mappings):
        dataset_name =  f"{mapping_name}.csv" if dataset_additional_text is None else f"{mapping_name}_{dataset_additional_text}.csv"
        with open(loc / dataset_name, "w") as out:
            csv_out=csv.writer(out)
            csv_out.writerow(["path", "label"])
            for row in zip(paths, labels_to_numeric(labels, mappings[mapping_name])):
                csv_out.writerow(row)


def load_depths(emb_dir, depths):
    data = {}
    print(emb_dir)
    for depth in depths:
        data[depth] = sorted(glob(os.path.join(emb_dir, depth, "*.jpeg")), key = lambda x: int(x.split(".")[-2].split("RUN")[-1]))
    return data

def tp_str_search(tp, tp_dict, num_tp):
    '''
    Tp=timepoints
    '''
    if str(tp) in tp_dict:
        return tp_dict[str(tp)]
    
    left_label = tp_str_search_left(tp, tp_dict)
    right_label = tp_str_search_right(tp, tp_dict, num_tp)
    
    infer_label = None
    if left_label is not None and right_label is not None:
        if left_label == right_label:
            infer_label = right_label
        else:
            print("CRITICAL: could not infer label of missing timepoints")     
            return None
    elif left_label is not None or right_label is not None:
        infer_label = left_label if left_label is not None else right_label
    else:
        print("CRITICAL: could not infer label of missing timepoints")
        return None
    
    if infer_label is not None:
        tp_dict[str(tp)] = infer_label
        return infer_label
    return None

        
def tp_str_search_left(tp, tp_dict):
    if str(tp) in tp_dict:
        return tp_dict[str(tp)]
    if tp<0:
        return None
    return tp_str_search_left(tp-1, tp_dict)

def tp_str_search_right(tp, tp_dict, num_tp):
    if str(tp) in tp_dict:
        return tp_dict[str(tp)]
    if tp >= num_tp:
        return None
    return tp_str_search_right(tp+1, tp_dict, num_tp)
    
def process_by_focal_depth(directory, output_dir, label_json, use_GPU=True, classes_to_use=None):
    '''
    Code used to generate the datasets for the data labeled by carson. Creates a new directory
    in data/interim that has subdirectories corresponding to all the timepoint labels
    '''
    TARGET_SIZE = (800, 800)
    output_dir = INTERIM_DATA_DIR / output_dir
    directory = RAW_DATA_DIR / directory
    label_json = RAW_DATA_DIR / label_json
    
    depths = ["F-45", "F-30", "F-15", "F0", "F15", "F30", "F45"]
    depths.reverse()
    LABEL_KEY = "timepoint_labels"
    
    if not os.path.exists(output_dir):
        #shutil.rmtree(output_dir)
        os.mkdir(output_dir)#shutil.rmtree(output_dir)
    
    with open(label_json, "r") as file:
        label_json = json.load(file)
    embs = [path for path in os.listdir(directory) if os.path.isdir(directory / path)]
    
    potential_labels = np.unique(list(label_json[embs[0]][LABEL_KEY].values()) + 
                                 list(label_json[embs[2]][LABEL_KEY].values()) + 
                                 list(label_json[embs[3]][LABEL_KEY].values())) # randomly sample three embryos to get all timepoints
    for potential_label in potential_labels:
        label_dir = os.path.join(output_dir, potential_label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
    
    model, device = load_faster_RCNN_model_device(use_GPU=use_GPU)
    
    for emb_dir in tqdm(embs):
        depth_ims = load_depths(directory / emb_dir, depths)
        
        labels_by_subj = label_json[emb_dir][LABEL_KEY]
        
        num_tp = len(depth_ims[depths[0]])
        for tp in tqdm(range(num_tp)):
            
            # for missing timepoints, interpolate if consistent before and after
            # timepoint before differs from timepoint after, skip
            tp_label = tp_str_search(tp, labels_by_subj, num_tp)
            if tp_label is None:
                continue
            
            if classes_to_use is not None and tp_label not in classes_to_use:
                continue
            
            fname_im_path = depth_ims[depths[-1]][tp]
            im_file = output_dir / tp_label / os.path.basename(fname_im_path).replace(".jpeg", ".tif")
            if os.path.exists(im_file):
                (f"Skipping {im_file}")
                continue
                #print("Found existing")
                #im_file = str(im_file).replace(".tif", "-1.tif")

            # save images as 7 x TARGET_SIZE images stacked along dim 0
            ims = []
            for depth in depths:
                fname = depth_ims[depth][tp]
                ims.append(imread(fname))
            ims = [focus_and_pad(im, TARGET_SIZE, model, device) for im in ims]
            ims = np.stack(ims, axis=0)
            assert(ims.shape == (7,800,800))
            imsave(im_file, ims)


def equalizeDistributionWithUnderSampling(paths, labels, max_num_per_class=None):
    '''
    Randomly samples max_num_per_class images from each class if there are more than max_num_per_class images in that class. Otherwise, 
    all images are kept. If max_num_per_class is None, it is set to the median number of images per class. 
    '''
    # Calculate the class distribution
    class_counts = Counter(labels)
    
    # If max_num_per_class is not provided, set it to the median number of images per class
    if max_num_per_class is None:
        max_num_per_class = int(np.median(list(class_counts.values())))
    
    # Separate paths and labels by class
    class_to_paths = {label: [] for label in class_counts}
    for path, label in zip(paths, labels):
        class_to_paths[label].append(path)
    
    # Under-sample classes
    balanced_paths = []
    balanced_labels = []
    for label, paths in class_to_paths.items():
        if len(paths) > max_num_per_class:
            sampled_paths = resample(paths, replace=False, n_samples=max_num_per_class, random_state=42)
        else:
            sampled_paths = paths
        balanced_paths.extend(sampled_paths)
        balanced_labels.extend([label] * len(sampled_paths))
    
    return balanced_paths, balanced_labels

            
if __name__ == "__main__":
    #process_by_focal_depth("Dataset2", "CarsonData1", "output2.json", use_GPU=True, classes_to_use=["tPNf", "t7", "t5", "t6", "t3"])
    paths, labels = get_all_image_paths_from_raws(loc = INTERIM_DATA_DIR / "CarsonData1", im_type="tif")
    
    # paths and labels are the paths to the images and their corresponding labels, respectively
    paths, labels = equalizeDistributionWithUnderSampling(paths, labels, max_num_per_class=1000)
    
    with open(RAW_DATA_DIR / "mappings.json", "r") as f:
        mappings = json.load(f)
    
    make_dataset(mappings, paths, labels, dataset_additional_text="undersampled-2")