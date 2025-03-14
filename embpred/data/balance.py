import os
import glob
import shutil
import random
from PIL import Image
from torchvision import transforms as v2
from typing import List, Optional, Callable, Tuple, Dict
import torch
from torchvision.io import read_image
from loguru import logger
from tqdm import tqdm
import numpy as np
from skimage.io import imsave

class DataBalancer:
    def __init__(
        self,
        img_paths: List[str],
        img_labels: List[int],
        T: Optional[int] = None,
        quartile: Optional[float] = 0.75,
        transforms: Optional[Callable] = None,
        oversample: bool = True,
        undersample: bool = True,
        aug_dir: Optional[str] = None,
        **extra_attrs  # arbitrary extra attributes as keyword arguments; each should be a list
    ):
        """
        Initializes the DataBalancer with image paths, labels, balancing parameters and extra attributes.
        
        Parameters:
          - img_paths (List[str]): List of image file paths.
          - img_labels (List[int]): Corresponding list of labels.
          - T (int, optional): Target number of images per class. If None, computed based on quartile.
          - quartile (float): The quartile for calculating T (default: 0.75).
          - transforms (Callable, optional): Transformation to apply for augmentation.
          - oversample (bool): Whether to perform oversampling.
          - undersample (bool): Whether to perform undersampling.
          - aug_dir (str, optional): Directory to save augmented images.
          - extra_attrs: Additional attribute lists. Each must have the same length as img_paths.
        """
        assert len(img_paths) == len(img_labels), "img_paths and img_labels must be the same length."
        self.original_img_paths = img_paths
        self.original_img_labels = img_labels

        # Verify extra attributes have the same length.
        self.original_extras: Dict[str, List] = {}
        for key, attr_list in extra_attrs.items():
            assert len(attr_list) == len(img_paths), f"Extra attribute '{key}' length must match img_paths"
            self.original_extras[key] = attr_list

        self.T = T
        self.transforms = transforms
        self.oversample = oversample
        self.undersample = undersample

        # Determine augmentation directory
        if aug_dir is None:
            base_dir = os.path.dirname(img_paths[0])
            self.aug_dir = os.path.join(base_dir, '.aug')
        else:
            self.aug_dir = aug_dir

        logger.info(f"Augmentation directory: {self.aug_dir}")
        os.makedirs(self.aug_dir, exist_ok=True)

        # Group data by class; for each sample, group image path and each extra attribute.
        self.class_to_imgs = {}
        self.class_to_extras = {}  # dict[label] = dict{ attr_name: list }
        for idx, (path, label) in enumerate(zip(self.original_img_paths, self.original_img_labels)):
            if label not in self.class_to_imgs:
                self.class_to_imgs[label] = []
                self.class_to_extras[label] = {key: [] for key in self.original_extras}
            self.class_to_imgs[label].append(path)
            for key, attr_list in self.original_extras.items():
                self.class_to_extras[label][key].append(attr_list[idx])
        
        # Calculate T if not provided
        if T is None:
            class_counts = self._calculate_class_counts(img_labels)
            self.T = int(np.percentile(list(class_counts.values()), quartile * 100))
            logger.info(f"Calculated target number of images per class (T) at {quartile} quartile: {self.T}")
        else:
            self.T = T

        # Prepare balanced data containers
        self.balanced_class_to_imgs = {}
        self.balanced_class_to_labels = {}
        self.balanced_class_to_extras = {}  # dict[label] = dict{ attr_name: list }
        self._balance()
    
    def _calculate_class_counts(self, labels: List[int]) -> dict:
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts

    def _balance(self):
        for label, imgs in tqdm(self.class_to_imgs.items(), desc="Balancing classes"):
            # Initialize balanced containers for this label
            self.balanced_class_to_imgs[label] = []
            self.balanced_class_to_labels[label] = []
            self.balanced_class_to_extras[label] = {key: [] for key in self.original_extras}
            
            num_imgs = len(imgs)
            
            # Undersampling: if there are more images than T
            if self.undersample and num_imgs > self.T:
                # Randomly select T indices for this class
                indices = random.sample(range(num_imgs), self.T)
            else:
                indices = list(range(num_imgs))
            
            # Add undersampled (or full) samples for this class
            for i in indices:
                self.balanced_class_to_imgs[label].append(self.class_to_imgs[label][i])
                self.balanced_class_to_labels[label].append(label)
                for key in self.original_extras:
                    self.balanced_class_to_extras[label][key].append(self.class_to_extras[label][key][i])
            
            # Oversampling: if less than T samples are present after undersampling
            if self.oversample and len(self.balanced_class_to_imgs[label]) < self.T:
                needed = self.T - len(self.balanced_class_to_imgs[label])
                augmentable_imgs = self.balanced_class_to_imgs[label].copy()
                # Also copy the corresponding extras
                augmentable_extras = {key: self.balanced_class_to_extras[label][key].copy() for key in self.original_extras}
                logger.info(f"Class {label} | Augmenting {needed} images starting from {len(augmentable_imgs)}")
                if not augmentable_imgs:
                    logger.warning(f"No images to augment for class {label}")
                    continue
                augmentation_tracker = {img: 0 for img in augmentable_imgs}
                augmented = 0
                img_count = len(augmentable_imgs)
                idx = 0

                while augmented < needed:
                    img = augmentable_imgs[idx % img_count]
                    idx += 1
                    augmentation_tracker[img] += 1
                    aug_number = augmentation_tracker[img]
                    base_name = os.path.splitext(os.path.basename(img))[0]
                    img_ext = os.path.splitext(img)[1]
                    aug_name = f"{base_name}-aug{aug_number}{img_ext}"
                    aug_path = os.path.join(self.aug_dir, aug_name)

                    image = read_image(img)
                    if self.transforms is None:
                        raise ValueError("Transforms must be provided for augmentation")
                    augmented_image = self.transforms(image)
                    # Permute and convert to numpy array
                    augmented_image = augmented_image.permute(1, 2, 0).numpy()
                    assert augmented_image.shape[2] == 3, "Augmented image must have 3 channels."
                    imsave(aug_path, augmented_image.astype('uint8'))
                    assert os.path.exists(aug_path), f"Augmented image not saved at {aug_path}"
                    
                    # Append augmented image and duplicate extras
                    self.balanced_class_to_imgs[label].append(aug_path)
                    self.balanced_class_to_labels[label].append(label)
                    for key in self.original_extras:
                        # Duplicate the attribute (use the same value as the original image that was augmented)
                        # Here we look-up the original extra corresponding to 'img' in augmentable_imgs.
                        # Note: If images are not unique, you may consider another strategy.
                        orig_value_index = self.class_to_imgs[label].index(img)
                        orig_extra = self.class_to_extras[label][key][orig_value_index]
                        self.balanced_class_to_extras[label][key].append(orig_extra)
                    augmented += 1

    def print_before_and_after(self):
        print(f"{'Class':<20}{'Before':<10}{'After':<10}")
        print("-" * 40)
        for label in self.class_to_imgs:
            before = len(self.class_to_imgs[label])
            after = len(self.balanced_class_to_imgs.get(label, []))
            print(f"{label:<20}{before:<10}{after:<10}")
    
    def balanced_img_paths(self) -> List[str]:
        balanced_paths = []
        for imgs in self.balanced_class_to_imgs.values():
            balanced_paths.extend(imgs)
        return balanced_paths

    def balanced_labels(self) -> List[int]:
        balanced_labels = []
        for labels in self.balanced_class_to_labels.values():
            balanced_labels.extend(labels)
        return balanced_labels

    def balanced_extras(self) -> Dict[str, List]:
        """Return a dictionary of extra attribute lists aggregated over all classes."""
        aggregated = {key: [] for key in self.original_extras}
        for label in self.balanced_class_to_extras:
            for key in self.original_extras:
                aggregated[key].extend(self.balanced_class_to_extras[label][key])
        return aggregated

    def delete_augmentation(self):
        shutil.rmtree(self.aug_dir)