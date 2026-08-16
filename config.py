# config.py
from pathlib import Path

# Project-level paths and shared configuration
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "static/images"

# Pixel-to-mm conversion factor used for radiographic measurements
PIXEL_TO_MM = 0.14

# Clinical reference threshold for VHS interpretation
BI_THRESHOLD = 10.7

# Detectron2 Mask R-CNN configuration for cardiothoracic ratio estimation
CTR_MODEL = {
    "config_path": BASE_DIR / "models/configs/config_5600_256_2_5e-3.yaml",
    "weights_path": BASE_DIR / "models/weights/model_5600_256_2_5e-3.pth",
    "score_thresh": 0.05,
    "classes": {0: "heart", 1: "thorax"}
}

# Detectron2 Mask R-CNN configuration for vertebral heart size estimation
BI_MODEL = {
    "config_path": BASE_DIR / "models/configs/config_4050_512_2_5e-3.yaml",
    "weights_path": BASE_DIR / "models/weights/model_4050_512_2_5e-3.pth",
    "score_thresh": 0.05,
    "classes": {0: "heart", 1: "t6", 2: "carina"}
}