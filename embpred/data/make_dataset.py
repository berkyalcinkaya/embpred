from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from glob import glob
import os
from typing import List, Union
from embpred.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR, INTERIM_DATA_DIR
from embpred.features import ExtractEmbFrame
import csv
import json
import torch
import cv2 
from skimage.io import imsave
import os
from PIL import Image
from torchvision import transforms

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
    def get_padding(image, target_size):
        return transforms.Pad(padding=(
            int((target_size[0] - image.width) / 2), 
            int((target_size[1] - image.height) / 2), 
            int((target_size[0] - image.width) / 2 + (target_size[0] - image.width) % 2), 
            int((target_size[1] - image.height) / 2 + (target_size[1] - image.height) % 2)
        ))

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

def make_dataset(mappings: List[dict], paths, labels, loc = PROCESSED_DATA_DIR):
    for mapping_name in tqdm(mappings):
        with open(loc / f"{mapping_name}.csv", "w") as out:
            csv_out=csv.writer(out)
            csv_out.writerow(["path", "label"])
            for row in zip(paths, labels_to_numeric(labels, mappings[mapping_name])):
                csv_out.writerow(row)

if __name__ == "__main__":
    mappings = RAW_DATA_DIR / "EmbStages/dataset1/mappings.json"
    assert (os.path.exists(mappings))
    with open(mappings, 'r') as json_file:
        mappings_dict = json.load(json_file)
    paths, labels = get_all_images_interim()
    make_dataset(mappings_dict, paths, labels)