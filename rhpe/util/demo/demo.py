from pathlib import Path
from typing import Optional

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
    title: Optional[str] = None,
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
    np.savez(
        output_dir.joinpath(f"{movie_path.stem}_keypoints_3d.npz"),
        keypoints_3d=keypoints_3d.numpy,
    )
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


def create_opunity(
    detector: KeyPointDetector,
    lifter: KeyPointLifter,
    frames: Frames,
    output_dir: Path,
):
    # Inference
    keypoints_2d = detector.detect_2d_keypoints(frames)
    normalized_keypoints_2d = normalize_keypoints(keypoints_2d)
    keypoints_3d = lifter.lift_up(normalized_keypoints_2d)
    save_as_opunity(keypoints_3d, output_dir)


class OPUnitySkeleton:
    hips = 0
    left_upper_leg = 1
    left_lower_leg = 2
    left_foot = 3
    right_upper_leg = 4
    right_lower_leg = 5
    right_foot = 6
    spine = 7
    chest = 8
    nect = 9
    head = 10
    right_upper_arm = 11
    right_lower_arm = 12
    right_hand = 13
    left_upper_arm = 14
    left_lower_arm = 15
    left_hand = 16


class RHPESkeleton:
    hips = 0
    left_upper_leg = 1
    left_lower_leg = 2
    left_foot = 3
    right_upper_leg = 4
    right_lower_leg = 5
    right_foot = 6
    spine = 7
    thorax = 8
    neck = 9
    head = 10
    right_shoulder = 11
    right_elbow = 12
    right_hand = 13
    left_shoulder = 14
    left_elbow = 15
    left_hand = 16


def save_as_opunity(keypoints_3d: KeyPoints3D, output_root: Path):
    SCALE = 1000
    correspondense = [
        RHPESkeleton.hips,
        RHPESkeleton.left_upper_leg,
        RHPESkeleton.left_lower_leg,
        RHPESkeleton.left_foot,
        RHPESkeleton.right_upper_leg,
        RHPESkeleton.right_lower_leg,
        RHPESkeleton.right_foot,
        RHPESkeleton.spine,
        RHPESkeleton.thorax,
        RHPESkeleton.neck,
        RHPESkeleton.head,
        RHPESkeleton.right_shoulder,
        RHPESkeleton.right_elbow,
        RHPESkeleton.right_hand,
        RHPESkeleton.left_shoulder,
        RHPESkeleton.left_elbow,
        RHPESkeleton.left_hand,
    ]
    kpts = keypoints_3d.numpy * SCALE
    kpts = kpts[:, correspondense]
    for i, frame in enumerate(kpts):
        target = frame.transpose([1, 0]).tolist()
        path = output_root.joinpath(f"{i}.txt")
        with path.open("w") as f:
            rows = [f'[{" ".join(map(str, arr))}]' for arr in target]
            opunity = f'[[{" ".join(rows)}]]'
            f.write(opunity)
