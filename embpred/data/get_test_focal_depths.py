from embpred.config import RAW_DATA_DIR, PROJ_ROOT
import json
import os
from glob import glob
import numpy as np
from skimage.io import imread, imsave
from embpred.features import load_faster_RCNN_model_device, ExtractEmbFrame, extract_emb_frame_2d
from skimage.transform import resize

output_dir = PROJ_ROOT / "reports" / "sample_focal_depths"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

embryo = "Subj_26790_Emb1"
mapping_file = RAW_DATA_DIR / "output.json"
image_dir = RAW_DATA_DIR / "DatasetNew" / embryo 

depths = ["F-45", "F-30", "F-15","F0", "F15", "F30", "F45"]
depth_ims = []
for depth in depths:
    depth_dir = image_dir / depth
    images = sorted(glob(os.path.join(depth_dir, "*.jpeg")), key = lambda x: int(x.split(".")[-2].split("RUN")[-1]))
    depth_ims.append(images[10])
    print(len(images))

for depth_im, depth in zip(depth_ims, depths):
    imsave(str(output_dir / depth) + ".jpeg", imread(depth_im))

# merge "F-15","F0", "F15" into an RGB image and save
imNeg15 = imread(depth_ims[2])
im0 = imread(depth_ims[3])
im15 = imread(depth_ims[4])

bbox_ims = []
for im in [im15, im0, imNeg15]:
    bbox_im = extract_emb_frame_2d(im)
    print(bbox_im.shape)
    bbox_im = resize(bbox_im, (224,224), anti_aliasing=True, preserve_range=True)
    bbox_ims.append(bbox_im)


# merge
merged_im = np.stack(bbox_ims, axis=2)
assert(merged_im.shape == (224,224,3))
imsave(str(output_dir / "merged_cropped_resized") + ".jpeg", merged_im) 

