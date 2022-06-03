from pathlib import Path
from gen_skes import load_model_layer
from rhpe.core import Frames, KeyPointDetector, KeyPointLifter, Renderer
import numpy as np
import cv2
from rhpe.keypoint_detector.rle_detector import RLEKeyPointDetector2D
from tools.inference import gen_pose
from tools.preprocess import revise_kpts, revise_skes
from tools.vis_h36m import render_animation

ROT = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)


def demo_movie(
    detector: KeyPointDetector,
    # lifter: KeyPointLifter,
    movie_path: Path,
    output_dir: Path,
):
    # # video = data_root + video
    # cap = cv2.VideoCapture(video)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = Frames.from_path(movie_path)

    # keypoints, scores = hrnet_pose(
    #     video, det_dim=416, num_peroson=num_person, gen_output=True
    # )
    # keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    # re_kpts = revise_kpts(keypoints, scores, valid_frames)
    # num_person = len(re_kpts)
    keypoints_2d = detector.detect_2d_keypoints(frames)
    re_kpts = revise_kpts(
        keypoints_2d.coordinates[np.newaxis],
        keypoints_2d.scores[np.newaxis],
        keypoints_2d.valid_frames[np.newaxis],
    )
    num_person = len(re_kpts)
    rf = 27
    # Loading 3D pose model
    model_pos = load_model_layer(rf)

    print("Generating 3D human pose ...")
    # pre-process keypoints

    pad = (rf - 1) // 2  # Padding on each side
    causal_shift = 0

    # Generating 3D poses
    prediction = gen_pose(
        re_kpts,
        keypoints_2d.valid_frames[np.newaxis],
        frames.width,
        frames.height,
        model_pos,
        pad,
        causal_shift,
    )

    prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])

    # If output two 3D human poses, put them in the same 3D coordinate system
    same_coord = False
    if num_person == 2:
        same_coord = True

    anim_output = {}
    for i, anim_prediction in enumerate(prediction):
        anim_output.update({"Reconstruction %d" % (i + 1): anim_prediction})

    viz_output = output_dir.joinpath(f"demo_{movie_path.stem}.mp4")
    print("Generating animation ...")
    # re_kpts: (M, T, N, 2) --> (T, M, N, 2)
    re_kpts = re_kpts.transpose(1, 0, 2, 3)
    render_animation(
        re_kpts,
        keypoints_2d.meta,
        anim_output,
        keypoints_2d.meta["skeleton"],
        25,
        30000,
        np.array(70.0, dtype=np.float32),
        str(viz_output),
        input_video_path=str(movie_path),
        viewport=(frames.width, frames.height),
        com_reconstrcution=same_coord,
    )


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
        for kpt in kpts:
            kpt = tuple(kpt.astype(np.int32))
            cv2.circle(frame, kpt, 10, (255, 0, 255))
        writer.write(frame)
    writer.release()


def main():
    device = "cuda:0"
    sandbox_path = Path("sandbox")
    movie_path = sandbox_path.joinpath("input", "khan_upperbody.mp4")
    output_dir = sandbox_path.joinpath("output")
    detector = RLEKeyPointDetector2D(device=device)
    # demo_movie_2d(detector, movie_path, output_dir)
    demo_movie(detector, movie_path, output_dir)


if __name__ == "__main__":
    main()
