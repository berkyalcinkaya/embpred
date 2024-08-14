from pathlib import Path
import numpy as np
import typer
from loguru import logger
import os
from tqdm import tqdm
import cv2
from glob import glob
from embpred.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR
import torch
from torchvision import transforms
from skimage.io import imread

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