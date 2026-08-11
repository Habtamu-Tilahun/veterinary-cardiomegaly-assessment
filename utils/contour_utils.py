import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def largest_contour(mask):
    """
    Extract the contour with the largest area from a binary mask.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def smooth_contour(contour, smoothing=1000):
    """
    Generate a smoothed contour using spline interpolation.
    """

    contour = contour.squeeze()

    if contour.ndim != 2 or len(contour) < 5:
        return contour.reshape(-1, 1, 2)

    x = contour[:, 0]
    y = contour[:, 1]

    tck, _ = splprep([x, y], s=smoothing)

    u = np.linspace(0, 1, len(contour))

    x_new, y_new = splev(u, tck)

    smooth = np.vstack([x_new, y_new]).T.astype(np.int32)

    return smooth.reshape(-1, 1, 2)