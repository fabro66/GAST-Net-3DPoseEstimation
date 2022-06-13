import cv2
import numpy as np
from common.camera import normalize_minmax_coordinates
from rhpe.core import KeyPoints2D
from rlepose.utils.transforms import _box_to_center_scale, get_affine_transform


class CropTransformer:
    def __init__(
        self, original_size: tuple[int, int], transformed_size: tuple[int, int]
    ):
        """cropping affine transformer

        Args:
            original_size (tuple[int, int]): (X, Y) original size of an image
            transformed_size (tuple[int, int]): (X, Y) transformed size of an image
        """
        self.original_size, self.transformed_size = original_size, transformed_size
        xmin, ymin, xmax, ymax = 0, 0, *original_size
        center, scale = _box_to_center_scale(
            xmin,
            ymin,
            xmax - xmin,
            ymax - ymin,
            transformed_size[0] / transformed_size[1],
            scale_mult=1.0,
        )
        self.matrix = get_affine_transform(
            center, scale, 0.0, (transformed_size[0], transformed_size[1])
        )
        self.matrix_square = np.concatenate(
            [self.matrix, np.array([[0.0, 0.0, 1.0]], dtype=self.matrix.dtype)], axis=0
        )
        self.inv_matrix = cv2.invertAffineTransform(self.matrix)
        self.inv_matrix_square = np.linalg.inv(self.matrix_square)

    def transform_image(self, src: np.ndarray) -> np.ndarray:
        """transform image from image space to neuralnet space

        Args:
            src (np.ndarray): H, W, 3

        Returns:
            np.ndarray: H', W', 3
        """
        return cv2.warpAffine(
            src,
            self.matrix,
            self.transformed_size,
            flags=cv2.INTER_LINEAR,
        )

    def inverse_image(self, src: np.ndarray) -> np.ndarray:
        """transform image from image space to neuralnet space

        Args:
            src (np.ndarray): H', W', 3

        Returns:
            np.ndarray: H, W, 3
        """
        return cv2.warpAffine(
            src,
            self.inv_matrix,
            self.original_size,
            flags=cv2.INTER_LINEAR,
        )

    def transform_position(self, positions: np.ndarray) -> np.ndarray:
        """transform positions in original space into position in neuralnet space

        Args:
            positions (np.ndarray): (F, 2) x, y coordinate

        Returns:
            np.ndarray: (F, 2) x, y coordinate
        """
        ones = np.ones((positions.shape[0], 1), dtype=positions.dtype)  # F, 1
        positions_3d = np.concatenate([positions, ones], axis=1)  # F, 3
        transformed_positions_3d = positions_3d @ self.matrix_square.transpose()  # F, 3
        transformed_position = transformed_positions_3d[:, 0:2]  # F, 2
        return transformed_position

    def inverse_position(self, positions: np.ndarray) -> np.ndarray:
        """transform position in neuralnet space into position in original space

        Args:
            positions (np.ndarray): (F, 2) x, y coordinate

        Returns:
            np.ndarray: (F, 2) x, y coordinate
        """
        ones = np.ones((positions.shape[0], 1), dtype=positions.dtype)  # F, 1
        positions_3d = np.concatenate([positions, ones], axis=1)  # F, 3
        transformed_positions_3d = (
            positions_3d @ self.inv_matrix_square.transpose()
        )  # F, 3
        transformed_position = transformed_positions_3d[:, 0:2]  # F, 2
        return transformed_position


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


def revise_kpts(
    h36m_kpts: np.ndarray, h36m_scores: np.ndarray, valid_frames: np.ndarray
) -> np.ndarray:
    """revise unconfident keypoints

    Args:
        h36m_kpts (np.ndarray): (F, J, 2)
        h36m_scores (np.ndarray): (F, J)
        valid_frames (np.ndarray): (F,)

    Returns:
        np.ndarray: (F, J, 2)
    """

    new_h36m_kpts = np.zeros_like(h36m_kpts)

    kpts = h36m_kpts[valid_frames]
    score = h36m_scores[valid_frames]

    # threshold_score = score > 0.3
    # if threshold_score.all():
    #     continue

    index_frame = np.where(np.sum(score < 0.3, axis=1) > 0)[0]

    for frame in index_frame:
        less_threshold_joints = np.where(score[frame] < 0.3)[0]

        intersect = [i for i in [2, 3, 5, 6] if i in less_threshold_joints]

        if [2, 3, 5, 6] == intersect:
            kpts[frame, [2, 3, 5, 6]] = kpts[frame, [1, 1, 4, 4]]
        elif [2, 3, 6] == intersect:
            kpts[frame, [2, 3, 6]] = kpts[frame, [1, 1, 5]]
        elif [3, 5, 6] == intersect:
            kpts[frame, [3, 5, 6]] = kpts[frame, [2, 4, 4]]
        elif [3, 6] == intersect:
            kpts[frame, [3, 6]] = kpts[frame, [2, 5]]
        elif [3] == intersect:
            kpts[frame, 3] = kpts[frame, 2]
        elif [6] == intersect:
            kpts[frame, 6] = kpts[frame, 5]
        else:
            continue

    new_h36m_kpts[valid_frames] = kpts
    return new_h36m_kpts


def revise_keypoints(keypoints_2d: KeyPoints2D) -> KeyPoints2D:
    revised_coordinates = revise_kpts(
        keypoints_2d.coordinates,
        keypoints_2d.scores,
        keypoints_2d.valid_frames,
    )
    return KeyPoints2D(
        coordinates=revised_coordinates,
        scores=keypoints_2d.scores,
        width=keypoints_2d.width,
        height=keypoints_2d.height,
        meta=keypoints_2d.meta,
        valid_frames=keypoints_2d.valid_frames,
    )


def normalize_keypoints(keypoints_2d: KeyPoints2D) -> KeyPoints2D:
    norm_kpts = normalize_minmax_coordinates(
        keypoints_2d.coordinates, w=keypoints_2d.width, h=keypoints_2d.height
    )
    return KeyPoints2D(
        coordinates=norm_kpts,
        scores=keypoints_2d.scores,
        width=keypoints_2d.width,
        height=keypoints_2d.height,
        meta=keypoints_2d.meta,
        valid_frames=keypoints_2d.valid_frames,
    )
