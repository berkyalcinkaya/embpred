import cv2
from embpred.config import RAW_DATA_DIR, PROJ_ROOT
import os
from glob import glob
from skimage.io import imread, imsave
from embpred.features import load_faster_RCNN_model_device, extract_emb_frame_2d
from skimage.transform import resize
from loguru import logger

logger.info("Running focal depth extraction script")
output_dir = PROJ_ROOT / "reports" / "sample_focal_depths"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

embryo = "Subj_26790_Emb1"
mapping_file = RAW_DATA_DIR / "output.json"
image_dir = RAW_DATA_DIR / "DatasetNew" / embryo 

depths = ["F-45", "F-30", "F-15","F0", "F15", "F30", "F45"]
depth_ims = []
for depth in depths:
    logger.info(f"Extracting focal depth {depth}")
    depth_dir = image_dir / depth
    images = sorted(glob(os.path.join(depth_dir, "*.jpeg")), key = lambda x: int(x.split(".")[-2].split("RUN")[-1]))
    depth_ims.append(images[10])
    logger.info(f"Found {len(images)} images for focal depth {depth}")

for depth_im, depth in zip(depth_ims, depths):
    imsave(str(output_dir / depth) + ".jpeg", imread(depth_im))
    logger.info(f"Saved image for focal depth {depth}")

# merge "F-15","F0", "F15" into an RGB image and save
imNeg15 = imread(depth_ims[2])
im0 = imread(depth_ims[3])
im15 = imread(depth_ims[4])

bbox_ims = []
logger.info("Extracting bounding boxes")
model, device = load_faster_RCNN_model_device()
for im, depth in [im15, im0, imNeg15], ["F15", "F0", "F-15"]:
    bbox_im = extract_emb_frame_2d(im, model, device)
    logger.info(f"Extracted bounding box of shape {bbox_im.shape}")
    bbox_im = resize(bbox_im, (224,224), anti_aliasing=True, preserve_range=True)
    logger.info(f"Resized bounding box to shape {bbox_im.shape}")

    # save bbox image
    imsave(str(output_dir / f"bbox_{depth}") + ".jpeg", bbox_im.astype('uint8'))
    logger.info(f"Saved bounding box image for focal depth {depth}")

    bbox_ims.append(bbox_im)


# merge
merged_im = cv2.merge(bbox_ims)
assert(merged_im.shape == (224,224,3))
logger.info(f"Merged bounding boxes of shape {merged_im.shape}")
merged_im_uint8 = merged_im.astype('uint8')
imsave(str(output_dir / "merged_cropped_resized") + ".jpeg", merged_im_uint8) 
logger.info(f"Saved merged image")

