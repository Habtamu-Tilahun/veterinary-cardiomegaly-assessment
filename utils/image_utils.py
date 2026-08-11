import cv2


def read_image(path):
    """
    Load an image from disk and validate that it was read successfully.
    """
        
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return image