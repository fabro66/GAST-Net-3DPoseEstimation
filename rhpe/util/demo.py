from pathlib import Path

import cv2
import numpy as np
from rhpe.core import Frames, KeyPointDetector, KeyPointLifter, KeyPoints2D, KeyPoints3D
from rhpe.util.visualize import render_animation


def render(
    frames: Frames,
    keypoints_2d: KeyPoints2D,
    keypoints_3d: KeyPoints3D,
    output_dir: Path,
):
    prediction = [keypoints_3d.numpy]
    prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])
    movie_path = frames.path

    output_path = output_dir.joinpath(f"demo_{movie_path.stem}.mp4")
    coordinates_2d = keypoints_2d.coordinates[np.newaxis]
    coordinates_2d = coordinates_2d.transpose(1, 0, 2, 3)
    render_animation(frames, keypoints_2d, keypoints_3d, output_path)


def demo_movie(
    detector: KeyPointDetector,
    lifter: KeyPointLifter,
    movie_path: Path,
    output_dir: Path,
):

    frames = Frames.from_path(movie_path)
    keypoints_2d = detector.detect_2d_keypoints(frames)
    keypoints_3d = lifter.lift_up(keypoints_2d)
    render(frames, keypoints_2d, keypoints_3d, output_dir)


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
