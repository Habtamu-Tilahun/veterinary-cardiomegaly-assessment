# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "static/images"

PIXEL_TO_MM = 0.14

BI_THRESHOLD = 10.7

CTR_MODEL = {
    "config_path": BASE_DIR / "models/configs/config_5600_256_2_5e-3.yaml",
    "weights_path": BASE_DIR / "models/weights/model_5600_256_2_5e-3.pth",
    "score_thresh": 0.05,
    "classes": {0: "heart", 1: "thorax"}
}

BI_MODEL = {
    "config_path": BASE_DIR / "models/configs/config_4050_256_2_5e-3.yaml",
    "weights_path": BASE_DIR / "models/weights/model_4050_256_2_5e-3.pth",
    "score_thresh": 0.05,
    "classes": {0: "heart", 1: "t4", 2: "carina"}
}