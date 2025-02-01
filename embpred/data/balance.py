import os
import glob
import shutil
import random
from PIL import Image
from torchvision import transforms as v2
from typing import List, Optional, Callable, Tuple
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
        quartile: float = 0.75,
        transforms: Optional[Callable] = None,
        oversample: bool = True,
        undersample: bool = True,
        aug_dir: Optional[str] = None
    ):
        """
        Initializes the DataBalancer with image paths, labels, and balancing parameters.

        Parameters:
        - img_paths (List[str]): List of image file paths.
        - img_labels (List[int]): Corresponding list of labels.
        - T (int, optional): Target number of images per class. If None, it will be calculated based on the specified quartile.
        - quartile (float): The quartile to use for calculating T if T is None. Default is 0.75.
        - transforms (Callable, optional): Transformations to apply for augmentation.
        - oversample (bool): Whether to perform oversampling.
        - undersample (bool): Whether to perform undersampling.
        - aug_dir (str, optional): Directory to save augmented images. Defaults to '.aug' in the first image's directory.
        """
        assert len(img_paths) == len(img_labels), "img_paths and img_labels must be the same length."
        self.original_img_paths = img_paths
        self.original_img_labels = img_labels
        self.transforms = transforms
        self.oversample = oversample
        self.undersample = undersample

        # Calculate T if not provided
        if T is None:
            class_counts = self._calculate_class_counts(img_labels)
            self.T = int(np.percentile(list(class_counts.values()), quartile * 100))
            logger.info(f"Calculated target number of images per class (T) at {quartile} quartile: {self.T}")
        else:
            self.T = T

        # Determine augmentation directory
        if aug_dir is None:
            base_dir = os.path.dirname(img_paths[0])
            self.aug_dir = os.path.join(base_dir, '.aug')
        else:
            self.aug_dir = aug_dir

        logger.info(f"Augmentation directory: {self.aug_dir}")
        os.makedirs(self.aug_dir, exist_ok=True)

    def _calculate_class_counts(self, labels: List[int]) -> dict:
        """
        Calculate the number of images per class.

        Parameters:
        - labels (List[int]): List of labels.

        Returns:
        - dict: Dictionary with class labels as keys and counts as values.
        """
        class_counts = {}
        for label in labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        return class_counts

    def _balance(self):
        for label, imgs in tqdm(self.class_to_imgs.items()):
            self.balanced_class_to_imgs[label] = []
            self.balanced_class_to_labels[label] = []
            num_imgs = len(imgs)

            # Undersampling
            if self.undersample and num_imgs > self.T:
                undersampled_imgs = random.sample(imgs, self.T)
                self.balanced_class_to_imgs[label].extend(undersampled_imgs)
                self.balanced_class_to_labels[label].extend([label]*self.T)
            else:
                self.balanced_class_to_imgs[label].extend(imgs)
                self.balanced_class_to_labels[label].extend([label]*num_imgs)

            # Oversampling
            if self.oversample and len(self.balanced_class_to_imgs[label]) < self.T:
                needed = self.T - len(self.balanced_class_to_imgs[label])
                augmentable_imgs = self.balanced_class_to_imgs[label].copy()
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
                    augmented_image = self.transforms(image)
                    augmented_image = augmented_image.permute(1, 2, 0).numpy()
                    assert augmented_image.shape[2] == 3, "Augmented image must have 3 channels."
                    imsave(aug_path, augmented_image.astype('uint8'))
                    assert os.path.exists(aug_path), f"Augmented image not saved at {aug_path}"
                    self.balanced_class_to_imgs[label].append(aug_path)
                    self.balanced_class_to_labels[label].append(label)
                    augmented += 1

    def print_before_and_after(self):
        """
        Prints the number of samples before and after balancing for each class.
        """
        print(f"{'Class':<20}{'Before':<10}{'After':<10}")
        print("-" * 40)
        for label in self.class_to_imgs:
            before = len(self.class_to_imgs[label])
            after = len(self.balanced_class_to_imgs.get(label, []))
            print(f"{label:<20}{before:<10}{after:<10}")
    
    
    def balanced_img_paths(self) -> List[str]:
        """
        Returns the list of balanced image paths.

        Returns:
        - List[str]: List of image file paths after balancing.
        """
        balanced_paths = []
        for imgs in self.balanced_class_to_imgs.values():
            balanced_paths.extend(imgs)
        return balanced_paths

    def balanced_labels(self) -> List[int]:
        """
        Returns the list of labels corresponding to balanced image paths.

        Returns:
        - List[int]: List of labels after balancing.
        """
        balanced_labels = []
        for labels in self.balanced_class_to_labels.values():
            balanced_labels.extend(labels)
        return balanced_labels

    def delete_augmentation(self):
        """
        Deletes the augmentation directory and its contents.
        """
        shutil.rmtree(self.aug_dir)