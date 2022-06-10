from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from common.camera import camera_to_world
from common.skeleton import Skeleton


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


@dataclass
class KeyPoints3D:
    numpy: np.ndarray
    meta: KeyPointsMeta

    def camera_to_world(self, rot: np.ndarray, t: int) -> np.ndarray:
        return camera_to_world(self.numpy, R=rot, t=t)
