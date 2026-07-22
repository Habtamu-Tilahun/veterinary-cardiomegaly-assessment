import cv2
import numpy as np

from config import PIXEL_TO_MM, BI_THRESHOLD


class BIService:
    """
    Buchanan Index (BI) estimation service.

    This implementation preserves the exact computational procedure used
    in the published work while improving code organization.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------

    def _get_best_prediction(self, pred_masks, pred_boxes, scores, class_indices):
        """
        Select the highest-confidence prediction for one class.

        This follows exactly the same logic as the original implementation.
        """

        class_indices_np = class_indices.cpu().numpy().flatten()

        masks = pred_masks[class_indices]
        scores_cls = scores[class_indices]

        if len(class_indices_np) > 1:
            boxes = pred_boxes.tensor.cpu().numpy()[class_indices_np]
        else:
            boxes = pred_boxes.tensor.cpu().numpy()[class_indices]

        if len(scores_cls) > 1:
            best_idx = scores_cls.argmax().item()
            mask = masks[best_idx]
            box = boxes[best_idx]
        else:
            mask = masks
            box = boxes

        mask = mask.squeeze().cpu().numpy().astype("uint8")

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        contour = contours[0]

        return mask, contour, box

    # ------------------------------------------------------------------

    def _get_best_carina_prediction(
        self,
        pred_masks,
        pred_boxes,
        scores,
        class_indices,
    ):
        """
        Special handling for the carina class.

        Kept identical to the original code.
        """

        class_indices_squeezed = class_indices.squeeze()

        masks = pred_masks[class_indices]

        if len(class_indices) == 1:
            boxes = pred_boxes[class_indices_squeezed.item()]
        else:
            boxes = pred_boxes[class_indices_squeezed]

        scores_cls = scores[class_indices]

        if len(scores_cls) > 1:
            best_idx = scores_cls.argmax().item()
            mask = masks[best_idx]
            box = boxes[best_idx]
        else:
            mask = masks
            box = boxes

        mask = mask.squeeze().cpu().numpy().astype("uint8")

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        contour = contours[0]

        box = box.tensor.cpu().numpy()[0]

        return mask, contour, box

    # ------------------------------------------------------------------

    @staticmethod
    def _get_intersection_point(heart_box, carina_box):
        """
        Compute the top-center point of the intersection between
        the heart and carina bounding boxes.

        Identical to the original implementation.
        """

        x1_h, y1_h, x2_h, y2_h = heart_box
        x1_c, y1_c, x2_c, y2_c = carina_box

        x1 = max(x1_h, x1_c)
        y1 = max(y1_h, y1_c)

        x2 = min(x2_h, x2_c)
        y2 = min(y2_h, y2_c)

        point_x = (x1 + x2) / 2
        point_y = y1

        return np.array([point_x, point_y])

    # ------------------------------------------------------------------

    @staticmethod
    def _find_apex(heart_contour, carina_point):
        """
        Find the contour point farthest from the carina point.

        Exact algorithm from the original implementation.
        """

        max_distance = 0
        apex = None

        for point in heart_contour:

            dist = np.linalg.norm(point[0] - carina_point)

            if dist > max_distance:
                max_distance = dist
                apex = point

        apex = np.array([apex[0][0], apex[0][1]])

        return apex, max_distance

    # ------------------------------------------------------------------

    @staticmethod
    def _get_minor_axis(perpendicular_slope, heart_contour):
        """
        Search for the longest contour segment whose slope matches
        the perpendicular slope.

        This is intentionally identical to the published implementation.
        """

        max_distance = 0

        pt1 = np.array([0, 0])
        pt2 = np.array([0, 0])

        for i in range(len(heart_contour)):

            x1, y1 = heart_contour[i][0]

            for j in range(len(heart_contour)):

                if i == j:
                    continue

                x2, y2 = heart_contour[j][0]

                if x2 == x1:
                    continue

                if x2 > x1:
                    slope = (y2 - y1) / (x2 - x1)
                else:
                    slope = (y1 - y2) / (x1 - x2)

                diff = abs(slope - perpendicular_slope)

                if diff < 0.001:

                    dist = np.linalg.norm(
                        np.array([x2, y2]) -
                        np.array([x1, y1])
                    )

                    if dist > max_distance:

                        max_distance = dist
                        pt1 = np.array([x1, y1])
                        pt2 = np.array([x2, y2])

        return pt1, pt2, max_distance

    def predict(self, image):
        """
        Estimate Buchanan Index.
        """

        # ----------------------------------------------------------
        # Run Detectron2 inference
        # ----------------------------------------------------------

        outputs = self.predictor(image)

        pred_classes = outputs["instances"].pred_classes
        pred_masks = outputs["instances"].pred_masks
        pred_boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores

        # ----------------------------------------------------------
        # Locate classes
        # ----------------------------------------------------------

        class_zero_indices = (pred_classes == 0).nonzero()
        class_one_indices = (pred_classes == 1).nonzero()
        class_two_indices = (pred_classes == 2).nonzero()

        if (
            len(class_zero_indices) == 0
            or len(class_one_indices) == 0
            or len(class_two_indices) == 0
        ):
            raise ValueError(
                "Heart, T4 or Carina could not be detected."
            )

        # ----------------------------------------------------------
        # Heart
        # ----------------------------------------------------------

        (
            heart_mask,
            heart_contour,
            heart_box,
        ) = self._get_best_prediction(
            pred_masks,
            pred_boxes,
            scores,
            class_zero_indices,
        )

        x1_heart, y1_heart, x2_heart, y2_heart = heart_box

        # ----------------------------------------------------------
        # Carina
        # ----------------------------------------------------------

        (
            carina_mask,
            carina_contour,
            carina_box,
        ) = self._get_best_carina_prediction(
            pred_masks,
            pred_boxes,
            scores,
            class_two_indices,
        )

        x1_carina, y1_carina, x2_carina, y2_carina = carina_box

        # ----------------------------------------------------------
        # Carina landmark
        # (identical to original implementation)
        # ----------------------------------------------------------

        p_carina = self._get_intersection_point(
            heart_box,
            carina_box,
        )

        point_x = p_carina[0]
        point_y = p_carina[1]

        # Uncomment for debugging if desired
        # cv2.circle(
        #     image,
        #     (int(point_x), int(point_y)),
        #     10,
        #     (0, 0, 255),
        #     -1,
        # )

        # ----------------------------------------------------------
        # Heart major axis
        # ----------------------------------------------------------

        p_apex, max_distance = self._find_apex(
            heart_contour,
            p_carina,
        )

        cv2.line(
            image,
            (int(point_x), int(point_y)),
            (int(p_apex[0]), int(p_apex[1])),
            (0, 0, 255),
            3,
        )

        heart_major_axis_length = (
            max_distance * PIXEL_TO_MM
        )

        # ----------------------------------------------------------
        # Heart minor axis
        # ----------------------------------------------------------

        slope = (
            (p_apex[1] - p_carina[1])
            /
            (p_apex[0] - p_carina[0])
        )

        slope = round(slope, 3)

        perp_slope = -1 / slope
        perp_slope = round(perp_slope, 3)

        (
            point1,
            point2,
            heart_minor_axis_length,
        ) = self._get_minor_axis(
            perp_slope,
            heart_contour,
        )

        cv2.line(
            image,
            (int(point1[0]), int(point1[1])),
            (int(point2[0]), int(point2[1])),
            (0, 0, 255),
            3,
        )

        heart_minor_axis_length *= PIXEL_TO_MM

        # ----------------------------------------------------------
        # T4
        # ----------------------------------------------------------

        (
            t4_mask,
            t4_contour,
            _
        ) = self._get_best_prediction(
            pred_masks,
            pred_boxes,
            scores,
            class_one_indices,
        )

        # ----------------------------------------------------------
        # Fit ellipse to T4
        # (identical to original implementation)
        # ----------------------------------------------------------

        t4_ellipse = cv2.fitEllipse(t4_contour)

        cv2.ellipse(
            image,
            t4_ellipse,
            (255, 255, 0),
            4,
        )

        # ----------------------------------------------------------
        # Major axis of fitted ellipse
        # ----------------------------------------------------------

        t4_major_axis_length = max(
            t4_ellipse[1][0],
            t4_ellipse[1][1],
        )

        t4_angle = t4_ellipse[2]
        t4_angle_rad = np.deg2rad(t4_angle - 90)

        t4_center = (
            int(t4_ellipse[0][0]),
            int(t4_ellipse[0][1]),
        )

        endpoint1 = (
            int(
                t4_center[0]
                + (t4_major_axis_length / 2)
                * np.cos(t4_angle_rad)
            ),
            int(
                t4_center[1]
                + (t4_major_axis_length / 2)
                * np.sin(t4_angle_rad)
            ),
        )

        endpoint2 = (
            int(
                t4_center[0]
                - (t4_major_axis_length / 2)
                * np.cos(t4_angle_rad)
            ),
            int(
                t4_center[1]
                - (t4_major_axis_length / 2)
                * np.sin(t4_angle_rad)
            ),
        )

        cv2.line(
            image,
            endpoint1,
            endpoint2,
            (0, 255, 0),
            4,
        )

        t4_major_axis_length *= PIXEL_TO_MM

        # ----------------------------------------------------------
        # Buchanan Index
        # ----------------------------------------------------------

        bi = (
            heart_major_axis_length
            + heart_minor_axis_length
        ) / t4_major_axis_length

        if bi > BI_THRESHOLD:
            finding = "Probably has cardiomegaly"
        else:
            finding = "No finding"

        # ----------------------------------------------------------
        # Draw contours
        # ----------------------------------------------------------

        cv2.drawContours(
            image,
            [heart_contour],
            -1,
            (255, 0, 0),
            5,
        )

        cv2.drawContours(
            image,
            [t4_contour],
            -1,
            (255, 0, 255),
            5,
        )

        cv2.drawContours(
            image,
            [carina_contour],
            -1,
            (0, 255, 255),
            5,
        )

        # ----------------------------------------------------------
        # Return identical outputs
        # ----------------------------------------------------------

        return {
            "Cardiac long axis": round(
                heart_major_axis_length,
                2,
            ),
            "Cardiac short axis": round(
                heart_minor_axis_length,
                2,
            ),
            "T4 size": round(
                t4_major_axis_length,
                2,
            ),
            "bi": round(
                bi,
                2,
            ),
            "finding": finding,
            "image": image,
        }