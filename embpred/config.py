from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from embpred.modeling.models import FirstNet2D

# Load environment variables from .env file if it exists
load_dotenv()

MODELS = {
    "FirstNet2D": FirstNet2D
}

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 11
EMB_OUTLIER_COUNT = 400 # remove embryos with less than 400 images, see 06-EDA notebook

RCNN_PATH = MODELS_DIR / "rcnn.pt"

TEMPORAL_MAP_PATH = PROCESSED_DATA_DIR / "timepoint_mapping.json"
MAP_PATH = RAW_DATA_DIR / "mappings.json"

default_transforms = []

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
