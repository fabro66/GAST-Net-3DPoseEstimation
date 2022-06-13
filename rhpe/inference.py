import numpy as np
from common.camera import camera_to_world
from common.generators import UnchunkedGenerator
from tools.inference import evaluate
from torch import nn

from rhpe.core import KeyPoints2D

ROT = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)


def gen_pose(
    keypoints_2d: KeyPoints2D, model_pos: nn.Module, pad: int, causal_shift: float = 0
) -> np.ndarray:
    skeleton = keypoints_2d.meta.skeleton
    gen = UnchunkedGenerator(
        None,
        None,
        keypoints_2d.coordinates[np.newaxis],
        pad=pad,
        causal_shift=causal_shift,
        augment=True,
        kps_left=skeleton.joints_left(),
        kps_right=skeleton.joints_right(),
        joints_left=None,
        joints_right=None,
    )
    prediction = evaluate(gen, model_pos)
    prediction_to_world = [camera_to_world(pred, R=ROT, t=0) for pred in prediction]

    return prediction_to_world[0]
