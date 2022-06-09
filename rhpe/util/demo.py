from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from rhpe.core import Frames, KeyPointDetector, KeyPointLifter, KeyPoints2D, KeyPoints3D
from tools.vis_h36m import render_animation


def render(
    keypoints_2d: KeyPoints2D,
    keypoints_3d: KeyPoints3D,
    movie_path: Path,
    output_dir: Path,
):
    prediction = [keypoints_3d.numpy]
    prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])

    same_coord = False

    anim_output = {}
    for i, anim_prediction in enumerate(prediction):
        anim_output.update({"Reconstruction %d" % (i + 1): anim_prediction})

    viz_output = output_dir.joinpath(f"demo_{movie_path.stem}.mp4")
    # re_kpts: (M, T, N, 2) --> (T, M, N, 2)
    coordinates_2d = keypoints_2d.coordinates[np.newaxis]
    coordinates_2d = coordinates_2d.transpose(1, 0, 2, 3)
    print("Generating animation ...")
    render_animation(
        coordinates_2d,
        asdict(keypoints_2d.meta),
        anim_output,
        keypoints_2d.meta.skeleton,
        25,
        30000,
        np.array(70.0, dtype=np.float32),
        str(viz_output),
        input_video_path=str(movie_path),
        viewport=(keypoints_2d.width, keypoints_2d.height),
        com_reconstrcution=same_coord,
    )


def demo_movie(
    detector: KeyPointDetector,
    lifter: KeyPointLifter,
    movie_path: Path,
    output_dir: Path,
):

    frames = Frames.from_path(movie_path)
    keypoints_2d = detector.detect_2d_keypoints(frames)
    keypoints_3d = lifter.lift_up(keypoints_2d)
    render(keypoints_2d, keypoints_3d, movie_path, output_dir)


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
