from embpred.config import INTERIM_DATA_DIR
import glob
import argparse
import logging
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as v2
import os

def balance_data(data_dir, cutoff):
    """
    Balances the dataset by ensuring each class has at least 'cutoff' number of images.
    Applies horizontal and vertical flips for augmentation until the cutoff is met.
    
    New images are saved with the format 'base_name-aug{x}.jpg', where x is the
    number of times the original image has been augmented.
    
    Parameters:
    - data_dir (str): Path to the main directory containing class subdirectories.
    - cutoff (int): Minimum number of images required per class.
    """
    transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
    ])
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    logging.info(f"Found {len(classes)} classes in '{data_dir}'.")

    for cls in tqdm(classes, desc="Balancing Classes"):
        cls_dir = os.path.join(data_dir, cls)
        image_paths = glob.glob(os.path.join(cls_dir, "*.*"))
        current_count = len(image_paths)
        logging.debug(f"Class '{cls}': {current_count} images found.")
        
        if current_count < cutoff:
            needed = cutoff - current_count
            augmented_count = 0
            idx = 0
            augmentation_tracker = {}
            logging.info(f"Balancing class '{cls}': {current_count} -> {cutoff} (need {needed} augmentations).")
            
            while augmented_count < needed and idx < len(image_paths) * 4:
                img_path = image_paths[idx % len(image_paths)]
                idx += 1
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Initialize augmentation count for the image
                if base_name not in augmentation_tracker:
                    augmentation_tracker[base_name] = 0
                
                # Limit to a maximum of 3 augmentations per image to prevent over-augmentation
                if augmentation_tracker[base_name] >= 3:
                    continue
                
                try:
                    img = Image.open(img_path)
                except Exception as e:
                    logging.warning(f"Failed to open image '{img_path}': {e}")
                    continue
                
                augmented_img = transform(img)
                augmentation_tracker[base_name] += 1
                aug_number = augmentation_tracker[base_name]
                aug_name = f"{base_name}-aug{aug_number}.jpg"
                aug_path = os.path.join(cls_dir, aug_name)
                
                if not os.path.exists(aug_path):
                    try:
                        augmented_img.save(aug_path)
                        augmented_count += 1
                        logging.debug(f"Saved augmented image '{aug_path}'.")
                    except Exception as e:
                        logging.warning(f"Failed to save augmented image '{aug_path}': {e}")
                
                # Prefer to augment more images vs multiple aug per image
                # So, iterate over images and augment each one as needed
                
            logging.info(f"Class '{cls}': {current_count} -> {current_count + augmented_count} images.")
    
    logging.info("Data balancing completed.")

def print_samples_per_class(data_dir: str) -> None:
    """
    Prints the number of samples per class in the dataset.

    Parameters:
    - data_dir (str): Path to the main directory containing class subdirectories.
    """
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    logging.info("Number of samples per class:")
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        image_paths = glob.glob(os.path.join(cls_dir, "*.*"))
        count = len(image_paths)
        logging.info(f"Class '{cls}': {count} samples")

def main():
    parser = argparse.ArgumentParser(description="Balance dataset by augmenting images.")
    parser.add_argument("directory", type=str, help="Name of the directory within INTERIM containing class subdirectories.")
    parser.add_argument("-d", "--dist", action="store_true", help="Print distribution of samples per class.")
    parser.add_argument("-t", "--threshold", type=int, help="Threshold T for minimum number of images per class.")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Construct full path
    data_dir = os.path.join(INTERIM_DATA_DIR, args.directory)
    
    if not os.path.isdir(data_dir):
        logging.error(f"The directory '{data_dir}' does not exist.")
        return
    
    if args.dist:
        print_samples_per_class(data_dir)
    else:
        if args.threshold is None:
            logging.error("Threshold T must be specified with -t when not using --dist.")
            return
        logging.info("Starting data balancing...")
        balance_data(data_dir, args.threshold)
        logging.info("Data balancing process finished.")
        print_samples_per_class(data_dir)

if __name__ == "__main__":
    main()