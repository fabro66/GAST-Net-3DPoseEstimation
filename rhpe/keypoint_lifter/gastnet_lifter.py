import numpy as np
from gen_skes import load_model_layer
from rhpe.core import KeyPointLifter, KeyPoints2D, KeyPoints3D
from tools.inference import gen_pose
from torch import nn


class GASTNetLifter(KeyPointLifter):
    def __init__(self, rf=27):
        assert rf in [27], "No supported receptive field for the model"
        self.rf = 27
        # Loading 3D pose model
        self.model_pos: nn.Module = load_model_layer(rf)

    def lift_up(self, keypoints_2d: KeyPoints2D) -> KeyPoints3D:
        pad = (self.rf - 1) // 2  # Padding on each side
        causal_shift = 0

        # Generating 3D poses
        prediction = gen_pose(
            keypoints_2d.coordinates[np.newaxis],
            keypoints_2d.valid_frames[np.newaxis],
            keypoints_2d.width,
            keypoints_2d.height,
            self.model_pos,
            pad,
            causal_shift,
        )
        return KeyPoints3D(prediction[0], keypoints_2d.meta)
