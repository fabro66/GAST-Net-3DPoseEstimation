from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
import cv2
import numpy as np
from common.camera import camera_to_world, normalize_screen_coordinates
from common.skeleton import Skeleton
from tools.mpii_coco_h36m import coco_h36m
from tools.visualization import render_animation
from abc import ABC, abstractmethod


class KeyPointDetector2D(ABC):
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

    @property
    def height(self) -> int:
        return self.numpy.shape[1]

    @property
    def width(self) -> int:
        return self.numpy.shape[2]

    @classmethod
    def from_path(cls, path: Path) -> Frames:
        raise NotImplementedError


class KeyPointsMeta(TypedDict):
    skeleton: Skeleton
    keypoints_symmetry: tuple[list[int], list[int]]
    layout_name: str
    num_joints: int


@dataclass
class KeyPoints2D:
    numpy: np.ndarray  # F, J, 2
    width: int
    height: int
    meta: KeyPointsMeta
    valid_frames: np.ndarray

    @classmethod
    def from_coco(cls, coco_numpy: np.ndarray, width: int, height: int) -> KeyPoints2D:
        """constructor from coco-formatted numpy

        Args:
            coco_numpy (np.ndarray): (F, 17, 2)

        Returns:
            KeyPoints2D: _description_
        """
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        keypoints, valid_frames = coco_h36m(coco_numpy)
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
            numpy=keypoints,
            width=width,
            height=height,
            meta=meta,
            valid_frames=valid_frames,
        )

    def as_input(self) -> np.ndarray:
        return normalize_screen_coordinates(self.numpy, self.width, self.height)


@dataclass
class KeyPoints3D:
    numpy: np.ndarray
    meta: KeyPointsMeta

    def camera_to_world(self, rot: np.ndarray, t: int) -> np.ndarray:
        return camera_to_world(self.numpy, R=rot, t=t)


class Renderer:
    def __init__(self, rot: np.ndarray):
        assert len(rot) == 4
        self.rot = rot

    def render(
        self,
        keypoints_2d: KeyPoints2D,
        keypoints_3d: KeyPoints3D,
        video_path: Path,
        output_path: Path,
    ):
        # We don't have the trajectory, but at least we can rebase the height
        prediction = keypoints_3d.camera_to_world(self.rot, t=0)
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        input_keypoints = keypoints_2d.as_input()
        valid_frames = keypoints_2d.valid_frames
        prediction_new = np.zeros((*input_keypoints.shape[:-1], 3), dtype=np.float32)
        prediction_new[valid_frames] = prediction
        anim_output = {"Reconstruction": prediction_new}

        # Get the width and height of video
        cap = cv2.VideoCapture(str(video_path))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        render_animation(
            keypoints_2d.numpy,
            keypoints_2d.meta,
            anim_output,
            keypoints_2d.meta["skeleton"],
            25,
            3000,
            np.array(70.0, dtype=np.float32),
            str(output_path),
            limit=-1,
            downsample=1,
            size=5,
            input_video_path=str(video_path),
            viewport=(width, height),
            input_video_skip=0,
        )
