import cv2
import numpy as np
from rlepose.utils.transforms import _box_to_center_scale, get_affine_transform


def crop_img(img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    assert len(bbox) == 4
    xmin, ymin, xmax, ymax = bbox

    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, (xmax - xmin) / (ymax - ymin), 1.0
    )
    transform = get_affine_transform(center, scale, 0.0, (xmax - xmin, ymax - ymin))
    cropped = cv2.warpAffine(
        img,
        transform,
        (xmax - xmin, ymax - ymin),
        flags=cv2.INTER_LINEAR,
    )
    return cropped
