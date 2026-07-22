# predictor.py
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class DetectronPredictor:

    def __init__(self, config_path: str, weights_path: str, score_thresh: float = 0.05):

        cfg = get_cfg()
        cfg.merge_from_file(config_path)

        cfg.MODEL.WEIGHTS = str(weights_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image):
        return self.predictor(image)