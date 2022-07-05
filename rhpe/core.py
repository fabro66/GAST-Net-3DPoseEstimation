from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from common.camera import camera_to_world
from common.skeleton import Skeleton
from tools.preprocess import h36m_coco_format
from typing_extensions import Self


class KeyPointDetector(ABC):
    @abstractmethod
    def detect_2d_keypoints(self, frames: Frames) -> KeyPoints2D:
        # Batchに分割するの忘れずに
        ...


class KeyPointLifter(ABC):
    @abstractmethod
    def lift_up(self, keypoints_2d: KeyPoints2D) -> KeyPoints3D:
        # Batchに分割するの忘れずに
        # GASTNetには最後のフレームのみ推論する方法があるかも
        ...


@dataclass
class Frames:
    numpy: np.ndarray  # F, H, W, 3
    path: Path
    fps: float

    @property
    def height(self) -> int:
        return self.numpy.shape[1]

    @property
    def width(self) -> int:
        return self.numpy.shape[2]

    @classmethod
    def from_path(cls, path: Path) -> Frames:
        cap = cv2.VideoCapture(str(path))

        assert cap.isOpened()
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        numpy = np.stack([cap.read()[1] for _ in range(video_len)])
        fps = cap.get(cv2.CAP_PROP_FPS)
        return cls(numpy, path, fps)


@dataclass
class KeyPointsMeta:
    skeleton: Skeleton
    keypoints_symmetry: tuple[list[int], list[int]]
    layout_name: str
    num_joints: int


@dataclass
class KeyPoints2D:
    coordinates: np.ndarray  # F, J, 2
    scores: np.ndarray  # F, J
    width: int
    height: int
    meta: KeyPointsMeta
    valid_frames: np.ndarray

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


@dataclass
class KeyPoints3D:
    numpy: np.ndarray
    meta: KeyPointsMeta

    def camera_to_world(self, rot: np.ndarray, t: int) -> np.ndarray:
        return camera_to_world(self.numpy, R=rot, t=t)

    def save_as_opunity(self, output_root: Path):
        SCALE = 1000

        kpts = self.numpy * SCALE
        for i, frame in enumerate(kpts):
            target = frame.transpose([1, 0]).tolist()
            path = output_root.joinpath(f"{i}.txt")
            with path.open("w") as f:
                rows = [f'[{" ".join(map(str, arr))}]' for arr in target]
                opunity = f'[[{" ".join(rows)}]]'
                f.write(opunity)
