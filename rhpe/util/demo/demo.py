from pathlib import Path

import cv2
import numpy as np
from rhpe.core import Frames, KeyPointDetector, KeyPointLifter, KeyPoints3D
from rhpe.util.transform import normalize_keypoints
from rhpe.util.visualize import KeyPoints2DAnimation, KeyPoints3DAnimation, Renderer


def demo_movie(
    detector: KeyPointDetector,
    lifter: KeyPointLifter,
    frames: Frames,
    output_dir: Path,
    expand: bool = True,
    title: str | None = None,
):

    # Inference
    keypoints_2d = detector.detect_2d_keypoints(frames)
    normalized_keypoints_2d = normalize_keypoints(keypoints_2d)
    keypoints_3d = lifter.lift_up(normalized_keypoints_2d)

    # Animation
    movie_path = frames.path
    output_path = output_dir.joinpath(f"demo_{movie_path.stem}.mp4")
    animation_2d = KeyPoints2DAnimation(keypoints_2d, frames, True, expand)
    animation_3d = KeyPoints3DAnimation(keypoints_3d, frames)
    animations = [
        animation_2d,
        animation_3d,
    ]
    renderer = Renderer(animations, title)
    renderer.render(output_path)


def demo_movie_2d(detector: KeyPointDetector, movie_path: Path, output_dir: Path):
    frames = Frames.from_path(movie_path)
    keypoints_2d = detector.detect_2d_keypoints(frames)
    basename = movie_path.stem
    filename = f"{basename}_demo.mp4"
    output_path = output_dir.joinpath(filename)

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, 24, (frames.width, frames.height)
    )
    for frame, kpts in zip(frames.numpy, keypoints_2d.coordinates):
        assert frame.shape == (frames.height, frames.width, 3)
        for kpt in kpts:
            kpt = tuple(kpt.astype(np.int32))
            cv2.circle(frame, kpt, 10, (255, 0, 255))
        writer.write(frame)
    writer.release()
