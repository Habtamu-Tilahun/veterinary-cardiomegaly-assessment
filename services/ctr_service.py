import cv2
import numpy as np

from config import PIXEL_TO_MM
from utils.contour_utils import (
    largest_contour,
    smooth_contour,
)


class CTRService:
    """
    Service responsible for estimating the Cardiothoracic Ratio
    (CTR) from thoracic radiographs and generating the associated
    measurements and visualization.
    """

    # Initialize the segmentation predictor.
    def __init__(self, predictor):
        self.predictor = predictor


    def _best_mask(self, classes, masks, scores, class_id):
        """
        Returns the highest-confidence mask for a given class.
        """

        idx = (classes == class_id).nonzero().flatten()

        if len(idx) == 0:
            raise ValueError(f"Class {class_id} not detected.")

        best = scores[idx].argmax().item()

        return masks[idx][best].cpu().numpy().astype("uint8")

 
    @staticmethod
    def _vertical_axis(contour, x_coordinate):
        """
        Finds the uppermost and lowermost contour points
        that share the same x-coordinate.
        """

        contour_points = contour[:, 0, :]

        vertical_points = contour_points[
            contour_points[:, 0] == x_coordinate
        ]

        if len(vertical_points) < 2:
            raise ValueError(
                "Unable to determine vertical measurement."
            )

        top = vertical_points[
            np.argmin(vertical_points[:, 1])
        ]

        bottom = vertical_points[
            np.argmax(vertical_points[:, 1])
        ]

        return top, bottom


    # Perform segmentation, anatomical measurements, CTR estimation, and visualization.
    def predict(self, image):

        outputs = self.predictor(image)

        pred_classes = outputs["instances"].pred_classes
        pred_masks = outputs["instances"].pred_masks
        scores = outputs["instances"].scores

        # =====================================================
        # Heart measurement
        # =====================================================

        heart_mask = self._best_mask(
            pred_classes,
            pred_masks,
            scores,
            class_id=0,
        )

        heart_contour = largest_contour(heart_mask)

        if heart_contour is None:
            raise ValueError("Heart contour not found.")

        x, y, w, h = cv2.boundingRect(heart_contour)

        heart_x = x + w // 2

        # Measure the heart through the center of its bounding box.
        heart_top, heart_bottom = self._vertical_axis(
            heart_contour,
            heart_x,
        )

        cv2.line(
            image,
            tuple(heart_top),
            tuple(heart_bottom),
            (255, 0, 255),
            5,
        )

        heart_size = (
            (heart_bottom[1] - heart_top[1])
            * PIXEL_TO_MM
        )

        # =====================================================
        # Thoracic measurement
        # =====================================================

        thorax_mask = self._best_mask(
            pred_classes,
            pred_masks,
            scores,
            class_id=1,
        )

        thorax_contour = largest_contour(thorax_mask)

        if thorax_contour is None:
            raise ValueError("Thorax contour not found.")

        thorax_x = heart_x + 25

        thorax_top, thorax_bottom = self._vertical_axis(
            thorax_contour,
            thorax_x,
        )

        cv2.line(
            image,
            tuple(thorax_top),
            tuple(thorax_bottom),
            (0, 255, 255),
            5,
        )

        thorax_size = (
            (thorax_bottom[1] - thorax_top[1])
            * PIXEL_TO_MM
        )

        # =====================================================
        # CTR estimation
        # =====================================================

        ctr = heart_size / thorax_size

        finding = (
            "Probably has cardiomegaly"
            if ctr >= (2 / 3)
            else "No finding"
        )

        # =====================================================
        # Visualization
        # =====================================================

        smoothed_heart = smooth_contour(
            heart_contour,
            25000.0,
        )

        smoothed_thorax = smooth_contour(
            thorax_contour,
            500000.0,
        )

        cv2.drawContours(
            image,
            [smoothed_heart],
            -1,
            (255, 255, 0),
            4,
        )

        cv2.drawContours(
            image,
            [smoothed_thorax],
            -1,
            (255, 0, 0),
            4,
        )

        # Rotate the annotated image for report visualization.
        image = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

        # =====================================================
        # Results
        # =====================================================

        return {
            "heart size": round(heart_size, 2),
            "thorax size": round(thorax_size, 2),
            "CTR": round(ctr, 3),
            "finding": finding,
            "image": image,
        }