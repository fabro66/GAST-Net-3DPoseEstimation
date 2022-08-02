import cv2
import numpy as np
from common.camera import normalize_minmax_coordinates
from rhpe.core import Frames, KeyPoints2D
from rlepose.utils.transforms import _box_to_center_scale, get_affine_transform

BBox = tuple[float, float, float, float]


class AffineTransformer:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def __init__(self, matrix: np.ndarray):
        assert matrix.shape == (2, 3)
        self.matrix = matrix
        self.matrix_square = np.concatenate(
            [self.matrix, np.array([[0.0, 0.0, 1.0]], dtype=self.matrix.dtype)], axis=0
        )
        self.inv_matrix = cv2.invertAffineTransform(self.matrix)
        try:
            self.inv_matrix_square = np.linalg.inv(self.matrix_square)
        except np.linalg.LinAlgError:
            print(
                "affine matrix is somehow singular. calling inverse operation will result in an error"
            )
            self.inv_matrix_square = None

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
        assert isinstance(self.inv_matrix_square, np.ndarray)
        ones = np.ones((positions.shape[0], 1), dtype=positions.dtype)  # F, 1
        positions_3d = np.concatenate([positions, ones], axis=1)  # F, 3
        transformed_positions_3d = (
            positions_3d @ self.inv_matrix_square.transpose()
        )  # F, 3
        transformed_position = transformed_positions_3d[:, 0:2]  # F, 2
        return transformed_position

    def transform_image(
        self,
        src: np.ndarray,
        size: tuple[int, int],
        border_color: tuple[int, int, int] = WHITE,
    ) -> np.ndarray:
        return cv2.warpAffine(
            src, self.matrix, size, flags=cv2.INTER_LINEAR, borderValue=border_color
        )

    def inverse_image(
        self,
        src: np.ndarray,
        size: tuple[int, int],
        border_color: tuple[int, int, int] = BLACK,
    ) -> np.ndarray:
        return cv2.warpAffine(
            src, self.inv_matrix, size, flags=cv2.INTER_LINEAR, borderValue=border_color
        )


class CropTransformer(AffineTransformer):
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
        matrix = get_affine_transform(
            center, scale, 0.0, (transformed_size[0], transformed_size[1])
        )
        super().__init__(matrix)

    def transform_image(self, src: np.ndarray) -> np.ndarray:
        """transform image from image space to neuralnet space

        Args:
            src (np.ndarray): H, W, 3

        Returns:
            np.ndarray: H', W', 3
        """
        return super().transform_image(src, self.transformed_size)

    def inverse_image(self, src: np.ndarray) -> np.ndarray:
        """transform image from image space to neuralnet space

        Args:
            src (np.ndarray): H', W', 3

        Returns:
            np.ndarray: H, W, 3
        """
        return super().inverse_image(src, self.original_size)


def crop_img(img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    assert len(bbox) == 4
    xmin, ymin, xmax, ymax = bbox

    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, (xmax - xmin) / (ymax - ymin), 1.0
    )
    transform = get_affine_transform(center, scale, 0.0, (xmax - xmin, ymax - ymin))
    cropped = cv2.warpAffine(
        img, transform, (xmax - xmin, ymax - ymin), flags=cv2.INTER_LINEAR,
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
        keypoints_2d.coordinates, keypoints_2d.scores, keypoints_2d.valid_frames,
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


def get_triangle(bbox: BBox) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    return np.array([[xmin, ymin], [xmax, ymax], [xmax, ymin]], dtype=np.float32)


def matrix_bbox_to_bbox(src_bbox: BBox, dst_bbox: BBox) -> np.ndarray:
    src_triangle = get_triangle(src_bbox)
    dst_triangle = get_triangle(dst_bbox)
    matrix = cv2.getAffineTransform(src_triangle, dst_triangle)
    return matrix


def get_minmax_bbox(keypoints_2d: KeyPoints2D) -> tuple[BBox, BBox]:
    coordinates = keypoints_2d.coordinates
    coordinate_minimum = np.min(coordinates, axis=(0, 1))
    minimum = np.minimum(coordinate_minimum, np.array([0.0, 0.0]))
    coordinate_maximum = np.max(coordinates, axis=(0, 1))
    maximum = np.maximum(
        coordinate_maximum, np.array([keypoints_2d.width, keypoints_2d.height])
    )
    src_bbox = (*minimum, *maximum)
    dst_bbox = (0, 0, *(maximum - minimum))  # type: ignore
    return src_bbox, dst_bbox


def expand_bbox(
    keypoints_2d: KeyPoints2D, frames: Frames
) -> tuple[KeyPoints2D, Frames]:
    """Transform keypoints and frames to visualize keypoints outside of images.
    Specifically, affine transform bbox (xmin, ymin, xmax, ymax) to (0, 0, xmax - xmin, ymax - ymin)
    where xmin = min(min(x_coordinates), 0) and xmax = max(max(x_coordinates), width) and so forth.

    Args:
        keypoints_2d (KeyPoints2D): original keypoints 2d to be visualized
        frames (Frames): original frames to be visualized

    Returns:
        tuple[KeyPoints2D, Frames]: affine transformed keypoints 2d and
        frames which include the entire keypoints
    """
    src_bbox, dst_bbox = get_minmax_bbox(keypoints_2d)
    transform_matrix = matrix_bbox_to_bbox(src_bbox, dst_bbox)
    transform = AffineTransformer(transform_matrix)
    coordinates = np.stack(
        [
            transform.transform_position(keypoints)
            for keypoints in keypoints_2d.coordinates
        ],
        axis=0,
    )
    new_keypoints_2d = KeyPoints2D(
        coordinates=coordinates,
        scores=keypoints_2d.scores,
        width=keypoints_2d.width,
        height=keypoints_2d.height,
        meta=keypoints_2d.meta,
        valid_frames=keypoints_2d.valid_frames,
    )
    size = (int(dst_bbox[2]), int(dst_bbox[3]))
    images = np.stack(
        [transform.transform_image(frame, size) for frame in frames.numpy], axis=0
    )
    new_frames = Frames(numpy=images, path=frames.path, fps=frames.fps)
    return new_keypoints_2d, new_frames


def to_rgb(frames: Frames) -> Frames:
    return Frames(
        numpy=np.stack(
            [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames.numpy]
        ),
        path=frames.path,
        fps=frames.fps,
    )
