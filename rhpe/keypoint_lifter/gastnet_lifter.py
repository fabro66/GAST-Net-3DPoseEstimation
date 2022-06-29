import numpy as np
import torch
from common.camera import camera_to_world
from common.generators import UnchunkedGenerator
from gen_skes import load_model_layer
from rhpe.core import KeyPointLifter, KeyPoints2D, KeyPoints3D
from tools.inference import evaluate
from torch import nn

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


class GASTNetLifter(KeyPointLifter):
    def __init__(self, rf=27, device: torch.device | None = None):
        assert rf in [27], f"{rf} is not supported receptive field for the model"
        self.rf = 27
        # Loading 3D pose model
        self.model_pos: nn.Module = load_model_layer(rf, device)

    def lift_up(self, keypoints_2d: KeyPoints2D) -> KeyPoints3D:
        pad = (self.rf - 1) // 2  # Padding on each side
        causal_shift = 0

        # Generating 3D poses
        prediction = gen_pose(
            keypoints_2d,
            self.model_pos,
            pad,
            causal_shift,
        )
        return KeyPoints3D(prediction, keypoints_2d.meta)
