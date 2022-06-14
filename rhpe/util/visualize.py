from abc import ABC, abstractmethod
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


class Animation(ABC):
    @abstractmethod
    def render_with_axes(self, step: int, ax: plt.Axes):
        ...

    @property
    @abstractmethod
    def num_frames(self) -> int:
        ...

    @property
    @abstractmethod
    def fps(self) -> int:
        ...

    @property
    @abstractmethod
    def projection(self) -> str:
        ...

    @abstractmethod
    def init_axes(self, axes: plt.Axes):
        ...


class KeyPoints2DAnimation(Animation):
    def __init__(
        self, keypoints_2d: KeyPoints2D, frames: Frames, background_frame: bool
    ):
        self.keypoints_2d = keypoints_2d
        self.frames = frames
        self.background_frame = background_frame

        # Rendering State
        self.initialized = False
        self.lines = list()
        self.points = None
        self.image = None

    @property
    def num_frames(self):
        return self.keypoints_2d.coordinates.shape[0]

    @property
    def fps(self):
        return self.frames.fps

    @property
    def projection(self) -> str:
        return "rectilinear"

    def init_axes(self, axes: plt.Axes):
        pass

    def initialize(self, ax: plt.Axes):
        step = 0
        joints_right_2d = self.keypoints_2d.meta.keypoints_symmetry[1]
        colors_2d = np.full(NUM_JOINTS, "black")
        colors_2d[joints_right_2d] = "red"
        if self.background_frame:
            self.image = ax.imshow(self.frames.numpy[step], aspect="equal")
        kpts2d_frame = self.keypoints_2d.coordinates[step]
        parents = self.keypoints_2d.meta.skeleton.parents()
        colors_2d = np.full(NUM_JOINTS, "black")
        colors_2d[joints_right_2d] = "red"
        # Draw 2D Points
        self.points = ax.scatter(
            *kpts2d_frame.T, MAKRER_SIZE, color=colors_2d, edgecolor=EDGECOLOR  # type: ignore
        )
        for joint, parent in zip(range(NUM_JOINTS), parents):
            if parent == -1:
                continue

            # Draw 2D Bones
            self.lines.append(
                ax.plot(
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

        self.initialized = True

    def update(self, step: int):
        if self.background_frame:
            self.image.set_data(self.frames.numpy[step])  # type: ignore
        kpts2d_frame = self.keypoints_2d.coordinates[step]
        parents = self.keypoints_2d.meta.skeleton.parents()
        # Change 2D Key Points
        self.points.set_offsets(kpts2d_frame)  # type: ignore

        for joint, parent in zip(range(NUM_JOINTS), parents):
            if parent == -1:
                continue

            # Change 2D Lines
            self.lines[joint - 1][0].set_data(
                [
                    kpts2d_frame[joint, 0],
                    kpts2d_frame[parent, 0],
                ],
                [
                    kpts2d_frame[joint, 1],
                    kpts2d_frame[parent, 1],
                ],
            )

    def render_with_axes(self, step: int, ax: plt.Axes):
        parents = self.keypoints_2d.meta.skeleton.parents()
        assert (
            len(parents) == 17 and self.keypoints_2d.meta.layout_name == "Human3.6M"
        ), "only support h36m keypoints format"
        if not self.initialized:
            self.initialize(ax)
        else:
            self.update(step)


class KeyPoints3DAnimation(Animation):
    def __init__(self, keypoints_3d: KeyPoints3D, frames: Frames):
        self.keypoints_3d = keypoints_3d
        self.frames = frames
        self.initialized = False
        self.lines = list()

    @property
    def num_frames(self):
        return self.keypoints_3d.numpy.shape[0]

    @property
    def fps(self):
        return self.frames.fps

    @property
    def projection(self) -> str:
        return "3d"

    def init_axes(self, ax: plt.Axes):
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlim3d([-RADIUS / 2, RADIUS / 2])
        ax.set_zlim3d([0, RADIUS])
        ax.set_ylim3d([-RADIUS / 2, RADIUS / 2])
        ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5  # type: ignore

    def initialize(self, ax):
        step = 0
        kpts3d_frame = self.keypoints_3d.numpy[step]
        parents = self.keypoints_3d.meta.skeleton.parents()
        for joint, parent in zip(range(NUM_JOINTS), parents):
            if parent == -1:
                continue
            # Draw 3D Bones
            col = h36m_color_edge(joint)
            self.lines.append(
                ax.plot(
                    [kpts3d_frame[joint, 0], kpts3d_frame[parent, 0]],
                    [kpts3d_frame[joint, 1], kpts3d_frame[parent, 1]],
                    [kpts3d_frame[joint, 2], kpts3d_frame[parent, 2]],
                    zdir="z",
                    c=col,
                    linewidth=LINEWIDTH,
                )
            )
        self.initialized = True

    def update(self, step: int):
        kpts3d_frame = self.keypoints_3d.numpy[step]
        parents = self.keypoints_3d.meta.skeleton.parents()
        for joint, parent in zip(range(NUM_JOINTS), parents):
            if parent == -1:
                continue

            # Change 3D Lines
            self.lines[joint - 1][0].set_xdata(
                [kpts3d_frame[joint, 0], kpts3d_frame[parent, 0]]
            )
            self.lines[joint - 1][0].set_ydata(
                [kpts3d_frame[joint, 1], kpts3d_frame[parent, 1]]
            )
            self.lines[joint - 1][0].set_3d_properties(
                [kpts3d_frame[joint, 2], kpts3d_frame[parent, 2]], zdir="z"
            )

    def render_with_axes(self, step: int, ax: plt.Axes):
        parents = self.keypoints_3d.meta.skeleton.parents()
        assert (
            len(parents) == 17 and self.keypoints_3d.meta.layout_name == "Human3.6M"
        ), "only support h36m keypoints format"
        if not self.initialized:
            self.initialize(ax)
        else:
            self.update(step)


class Renderer:
    def __init__(self, animations: list[Animation]):
        assert len(animations) > 0, "no annimation registered"
        assert all(
            [anim.num_frames == animations[0].num_frames for anim in animations]
        ), "number of frames of all animations must be the same"
        assert all(
            [anim.fps == animations[0].fps for anim in animations]
        ), "fps of all animations must be the same"
        self.fps = animations[0].fps
        self.num_frames = animations[0].num_frames
        self.animations = animations
        self.fig = plt.figure()
        self.axes = [
            self.fig.add_subplot(
                1, len(animations), idx + 1, projection=animation.projection
            )
            for idx, animation in enumerate(animations)
        ]
        for ax, animation in zip(self.axes, animations):
            animation.init_axes(ax)

    def render(self, output_path: Path):
        def update(step: int):
            for ax, animation in zip(self.axes, self.animations):
                animation.render_with_axes(step, ax)

        anim = FuncAnimation(self.fig, update, frames=tqdm(range(self.num_frames)))  # type: ignore

        if output_path.suffix == ".mp4":
            Writer = writers["ffmpeg"]
            writer = Writer(fps=self.fps, metadata={}, bitrate=BITRATE)
            anim.save(str(output_path), writer=writer)
        else:
            raise ValueError("Unsupported output format (only .mp4 is supported)")
        plt.close()
