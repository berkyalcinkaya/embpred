import os
import glob
import shutil
import argparse
from tqdm import tqdm
import logging

def get_subdirs(directory):
    return [os.path.join(directory, sdir) for sdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, sdir))]

def count_subdirs(directory):
    return len(get_subdirs(directory))

def main(source_dir, destination_dir=None):
    if destination_dir is not None:
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        target_dir = destination_dir
    else:
        target_dir = source_dir

    logging.info(f"Processing directory: {source_dir}")
    if destination_dir:
        logging.info(f"Output will be saved to: {destination_dir}")

    subdirs = glob.glob(f"{source_dir}/*")
    for subdir in tqdm(subdirs, desc="Reorganizing directories"):
        if os.path.isdir(subdir):
            try:
                if count_subdirs(subdir) != 7:
                    subsubdirs = get_subdirs(subdir)
                    for subsubdir in subsubdirs:
                        new_name = f"{os.path.basename(subdir)}_{os.path.basename(subsubdir)}"
                        new_path = os.path.join(target_dir, new_name)
                        shutil.move(subsubdir, new_path)
                        logging.info(f"Moved {subsubdir} to {new_path}")
                    shutil.rmtree(subdir)  # Remove the original directory and its contents
                    logging.info(f"Removed original directory: {subdir}")
            except Exception as e:
                logging.error(f"Error processing {subdir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize directory structure.")
    parser.add_argument("source_dir", type=str, help="Path to the source directory to be reorganized.")
    parser.add_argument("--destination", type=str, default=None, help="Optional path to save the reorganized directory structure. If not specified, the original directory will be modified.")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args.source_dir, args.destination)
