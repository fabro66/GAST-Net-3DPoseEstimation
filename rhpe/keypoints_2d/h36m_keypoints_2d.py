import numpy as np
from common.skeleton import Skeleton
from rhpe.core import KeyPoints2D, KeyPointsMeta
from rhpe.util.transform import revise_kpts
from tools.preprocess import h36m_coco_format
from typing_extensions import Self


class H36MKeyPoints2D(KeyPoints2D):
    @classmethod
    def from_coco(
        cls, coordinates: np.ndarray, scores: np.ndarray, width: int, height: int
    ) -> Self:
        """constructor from coco-formatted numpy

        Args:
            coordinate (np.ndarray): (F, J, 2)
            scores (np.ndarray): (F, )

        Returns:
            KeyPoints2D: H36M KeyPoints2D Object
        """
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        coordinates, scores, valid_frames = h36m_coco_format(
            coordinates[np.newaxis], scores[np.newaxis]
        )
        skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            joints_left=[4, 5, 6, 11, 12, 13],
            joints_right=[1, 2, 3, 14, 15, 16],
        )
        meta = KeyPointsMeta(
            skeleton=skeleton,
            keypoints_symmetry=(joints_left, joints_right),
            layout_name="Human3.6M",
            num_joints=17,
        )
        return cls(
            coordinates=coordinates[0],
            scores=scores[0],
            width=width,
            height=height,
            meta=meta,
            valid_frames=valid_frames[0],
        )


class RevisedH36MKeyPoints2D(KeyPoints2D):
    @classmethod
    def from_h36m(cls, h36m: H36MKeyPoints2D) -> Self:
        revised_coordinates = revise_kpts(
            h36m.coordinates,
            h36m.scores,
            h36m.valid_frames,
        )
        return cls(
            coordinates=revised_coordinates,
            scores=h36m.scores,
            width=h36m.width,
            height=h36m.height,
            meta=h36m.meta,
            valid_frames=h36m.valid_frames,
        )
