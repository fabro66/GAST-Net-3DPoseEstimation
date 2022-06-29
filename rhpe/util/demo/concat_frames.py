from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from rhpe.core import Frames
from rhpe.keypoint_detector.rle_detector import RLEKeyPointDetector2D
from rhpe.keypoint_lifter.gastnet_lifter import GASTNetLifter
from rhpe.util.demo.demo import demo_movie


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


def vis_linear(start_ratio: float, end_ratio: float, split: int):
    assert split > 1
    # Setup Path
    midterm_dir = Path().joinpath("sandbox", "midterm")
    demo_dir = midterm_dir.joinpath("demo")
    concat_dir = midterm_dir.joinpath("concat")
    original_files = list(midterm_dir.joinpath("original").iterdir())
    assert len(original_files) == 1
    original_path = original_files[0]
    concat_path = concat_dir.joinpath(f"concat_{original_path.stem}.mp4")

    # Crop
    original_frames = Frames.from_path(original_path)
    ratios = [
        start_ratio + (end_ratio - start_ratio) * i / (split - 1) for i in range(split)
    ]
    frames_list = [
        Frames(
            crop_frames(original_frames.numpy, ratio),
            demo_dir.joinpath(f"{original_path.stem}_{ratio:.2f}.mp4"),
            original_frames.fps,
        )
        for ratio in ratios
    ]

    # Execute estimation and save demo videos
    device = torch.device("cuda:3")
    detector = RLEKeyPointDetector2D(device=device)
    lifter = GASTNetLifter(rf=27, device=device)
    for frames, ratio in zip(frames_list, ratios):
        demo_movie(
            detector,
            lifter,
            frames,
            demo_dir,
            expand=False,
            title=f"Cut: {int((1 - ratio) * 100)}%",
        )

    # Concate Frames in demo_dir
    concat_frames_list = [
        Frames.from_path(path) for path in sorted(demo_dir.iterdir(), reverse=True)
    ]
    concat_frms = concat_frames(concat_frames_list, concat_path)
    save_frames(concat_frms)
