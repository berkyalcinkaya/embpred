from embpred.config import RAW_DATA_DIR, PROJ_ROOT
import json
import os
from glob import glob
import numpy as np
from skimage.io import imread, imsave

output_dir = PROJ_ROOT / "reports" / "sample_images"
embryo = "Subj_26790_Emb1"
mapping_file = RAW_DATA_DIR / "output.json"
image_dir = RAW_DATA_DIR / "DatasetNew" / embryo / "F0"
print(os.path.exists(image_dir))
images = sorted(glob(os.path.join(image_dir, "*.jpeg")), key = lambda x: int(x.split(".")[-2].split("RUN")[-1]))
print(len(images))

with open(mapping_file) as f:
    mapping = json.load(f)
embryo_map = mapping[embryo]

labels = embryo_map.keys()
unique_labels = np.unique(list(labels))

for i, image_path in  enumerate(images):
    image_name = os.path.basename(image_path)
    if str(i) not in embryo_map:
        continue
    label = embryo_map[str(i)]
    if label in unique_labels:
        unique_labels = np.delete(unique_labels, np.where(unique_labels == label))
        print(f"Image {i} has label {label}")
        print(f"Unique labels left: {unique_labels}")
        image = imread(image_path)
        imsave(str(output_dir / label) + ".jpeg", image)
    if len(unique_labels) == 0:
        break


