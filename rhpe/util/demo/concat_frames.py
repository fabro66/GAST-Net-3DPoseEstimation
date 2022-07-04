from copy import deepcopy
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from rhpe.core import Frames, KeyPointDetector, KeyPointLifter
from rhpe.keypoint_detector.hrnet_detector import HRNetDetector
from rhpe.keypoint_detector.rle_detector import RLEKeyPointDetector2D
from rhpe.keypoint_lifter.gastnet_lifter import GASTNetLifter
from rhpe.util.transform import normalize_keypoints
from rhpe.util.visualize import (
    Animation,
    KeyPoints2DAnimation,
    KeyPoints3DAnimation,
    Renderer,
)


def crop_frames(frames: np.ndarray, height_ratio: float) -> np.ndarray:
    height = int(frames.shape[1] * height_ratio)
    new_frames = frames[:, :height]
    return new_frames


def assert_common_properties(frames_list: Sequence[Frames]):
    fps = frames_list[0].fps
    width = frames_list[0].width
    assert all([frames.fps == fps for frames in frames_list])
    assert all([frames.width == width for frames in frames_list])


def concat_frames(frames_list: Sequence[Frames], output_path: Path) -> Frames:
    assert_common_properties(frames_list)
    frms_np_list = [frames.numpy for frames in frames_list]
    new_np = np.concatenate(frms_np_list, axis=1)
    fps = frames_list[0].fps
    new_frames = Frames(new_np, output_path, fps)
    return new_frames


def save_frames(frames: Frames):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        str(frames.path), fourcc, frames.fps, (frames.width, frames.height)
    )
    for img in frames.numpy:
        video.write(img)
    video.release()


def animation_linear(
    start_ratio: float,
    end_ratio: float,
    split: int,
    detector: KeyPointDetector,
    lifter: KeyPointLifter,
    original_path: Path,
) -> list[Animation]:
    assert split > 1
    # Crop
    original_frames = Frames.from_path(original_path)
    ratios = [
        start_ratio + (end_ratio - start_ratio) * i / (split - 1)
        for i in reversed(range(split))
    ]
    frames_list = [
        Frames(
            crop_frames(original_frames.numpy, ratio),
            original_frames.path,
            original_frames.fps,
        )
        for ratio in ratios
    ]

    original_size = (original_frames.height, original_frames.width)
    # Execute estimation and save demo videos
    animations = (
        np.array(
            [
                create_animation(detector, lifter, frames, original_size)
                for frames in frames_list
            ]
        )
        .T.reshape(-1)
        .tolist()
    )
    return animations


def vis_linear_comparison(
    start: float,
    end: float,
    split: int,
    root_dir: Path,
    device: torch.device | None = torch.device("cuda:0"),
):
    # Setup Path
    original_files = list(root_dir.joinpath("original").iterdir())
    assert len(original_files) == 1
    original_path = original_files[0]
    output_path = root_dir.joinpath("concat", f"concat_{original_path.stem}.mp4")

    rle_detector = RLEKeyPointDetector2D(device)
    hrnet_detector = HRNetDetector(device)
    lifter = GASTNetLifter(rf=27, device=device)
    rle_animations = animation_linear(
        start, end, split, rle_detector, lifter, original_path
    )
    hrnet_animations = animation_linear(
        start, end, split, hrnet_detector, lifter, original_path
    )
    animations = rle_animations + hrnet_animations
    renderer = Renderer(animations, None, (4, split))
    renderer.render(output_path)


def create_animation(
    detector: KeyPointDetector,
    lifter: KeyPointLifter,
    frames: Frames,
    original_size: tuple[int, int],
) -> tuple[Animation, Animation]:
    # Inference
    keypoints_2d = detector.detect_2d_keypoints(frames)
    normalized_keypoints_2d = normalize_keypoints(keypoints_2d)
    keypoints_3d = lifter.lift_up(normalized_keypoints_2d)

    # Animation
    frames = pad_with_white(frames, original_size)
    animation_2d = KeyPoints2DAnimation(
        keypoints_2d, frames, background_frame=True, expand=False
    )
    animation_3d = KeyPoints3DAnimation(keypoints_3d, frames)
    return animation_2d, animation_3d


def pad_with_white(source: Frames, size: tuple[int, int]):
    height, width = size
    target = deepcopy(source)
    if target.height < height:
        padding = (
            np.ones(
                (target.numpy.shape[0], height - target.height, target.width, 3),
                dtype=source.numpy.dtype,
            )
            * 255
        )
        target.numpy = np.concatenate([target.numpy, padding], axis=1)
    if target.width < width:
        padding = (
            np.ones(
                (target.numpy.shape[0], width - target.width, target.height, 3),
                dtype=source.numpy.dtype,
            )
            * 255
        )
        target.numpy = np.concatenate([target.numpy, padding], axis=2)
    return target
