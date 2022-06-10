from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers  # type: ignore
from rhpe.core import Frames, KeyPoints2D, KeyPoints3D
from tools.color_edge import h36m_color_edge
from tqdm import tqdm

RADIUS = 2.5

ELEV = 15.0
AZIM = 70.0
NUM_JOINTS = 17
LINEWIDTH = 3
EDGECOLOR = "white"
MAKRER_SIZE = 10
BITRATE = 30000


def render_animation(
    frames: Frames,
    keypoints_2d: KeyPoints2D,
    keypoints_3d: KeyPoints3D,
    output_path: Path,
):
    plt.ioff()

    # ax 2D
    fig = plt.figure()
    ax_2d = fig.add_subplot(1, 2, 1)

    # ax 3D
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_3d.view_init(elev=ELEV, azim=AZIM)
    ax_3d.set_xlim3d([-RADIUS / 2, RADIUS / 2])
    ax_3d.set_zlim3d([0, RADIUS])
    ax_3d.set_ylim3d([-RADIUS / 2, RADIUS / 2])
    ax_3d.set_aspect("auto")
    ax_3d.set_xticklabels([])
    ax_3d.set_yticklabels([])
    ax_3d.set_zticklabels([])
    ax_3d.dist = 7.5

    initialized = False
    image = None
    points = None
    lines = list()
    lines_3d = list()

    num_frames = len(frames.numpy)

    def update_video(frame_idx: int):
        nonlocal initialized, image, points
        joints_right_2d = keypoints_2d.meta.keypoints_symmetry[1]

        colors_2d = np.full(NUM_JOINTS, "black")
        colors_2d[joints_right_2d] = "red"
        kpts2d_frame = keypoints_2d.coordinates[frame_idx]
        kpts3d_frame = keypoints_3d.numpy[frame_idx]
        parents = keypoints_2d.meta.skeleton.parents()
        assert (
            len(parents) == 17 and keypoints_2d.meta.layout_name == "Human3.6M"
        ), "only support h36m keypoints format"
        if not initialized:
            image = ax_2d.imshow(frames.numpy[frame_idx], aspect="equal")
            # Draw 2D Points
            points = ax_2d.scatter(
                *kpts2d_frame.T, MAKRER_SIZE, color=colors_2d, edgecolor=EDGECOLOR
            )
            for joint, parent in zip(range(NUM_JOINTS), parents):
                if parent == -1:
                    continue

                # Draw 2D Bones
                lines.append(
                    ax_2d.plot(
                        [
                            kpts2d_frame[joint, 0],
                            kpts2d_frame[parent, 0],
                        ],
                        [
                            kpts2d_frame[joint, 1],
                            kpts2d_frame[parent, 1],
                        ],
                        color="pink",
                    )
                )

                # Draw 3D Bones
                col = h36m_color_edge(joint)
                lines_3d.append(
                    ax_3d.plot(
                        [kpts3d_frame[joint, 0], kpts3d_frame[parent, 0]],
                        [kpts3d_frame[joint, 1], kpts3d_frame[parent, 1]],
                        [kpts3d_frame[joint, 2], kpts3d_frame[parent, 2]],
                        zdir="z",
                        c=col,
                        linewidth=LINEWIDTH,
                    )
                )
            initialized = True
        else:
            image.set_data(frames.numpy[frame_idx])  # type: ignore

            # Change 2D Key Points
            points.set_offsets(kpts2d_frame)  # type: ignore

            for joint, parent in zip(range(NUM_JOINTS), parents):
                if parent == -1:
                    continue

                # Change 2D Lines
                lines[joint - 1][0].set_data(
                    [
                        kpts2d_frame[joint, 0],
                        kpts2d_frame[parent, 0],
                    ],
                    [
                        kpts2d_frame[joint, 1],
                        kpts2d_frame[parent, 1],
                    ],
                )

                # Change 3D Lines
                lines_3d[joint - 1][0].set_xdata(
                    [kpts3d_frame[joint, 0], kpts3d_frame[parent, 0]]
                )
                lines_3d[joint - 1][0].set_ydata(
                    [kpts3d_frame[joint, 1], kpts3d_frame[parent, 1]]
                )
                lines_3d[joint - 1][0].set_3d_properties(
                    [kpts3d_frame[joint, 2], kpts3d_frame[parent, 2]], zdir="z"
                )

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=tqdm(range(num_frames)))  # type: ignore
    if output_path.suffix == ".mp4":
        Writer = writers["ffmpeg"]
        writer = Writer(fps=frames.fps, metadata={}, bitrate=BITRATE)
        anim.save(str(output_path), writer=writer)
    else:
        raise ValueError("Unsupported output format (only .mp4 is supported")
    plt.close()
