from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
from lib.pose.hrnet.lib.utils.transforms import transform_preds
from lib.pose.hrnet.pose_estimation.gen_kpts import detect_bbox
from rhpe.core import Frames, KeyPointDetector, KeyPoints2D
from rlepose.models import builder  # type: ignore
from torch import nn
from tqdm import tqdm


class RLEModelOutput(Protocol):
    pred_jts: Sequence[torch.Tensor]
    maxvals: Sequence[torch.Tensor]


class RLEKeyPointDetector2D(KeyPointDetector):
    def __init__(
        self,
        ckpt_path: Path = Path().joinpath("pretrained_model", "coco-laplace-rle.pth"),
        device: Optional[torch.device] = torch.device("cuda:0"),
    ):
        # Only support this input size for now.
        self.input_height, self.input_width = 256, 192
        self.device = device
        self.model = self.load_model(ckpt_path, device)

    def load_model(
        self, model_path: Path, device: Optional[torch.device],
    ) -> nn.Module:
        model = builder.build_sppe(
            {
                "TYPE": "RegressFlow",
                "PRETRAINED": "",
                "TRY_LOAD": "",
                "NUM_FC_FILTERS": [-1],
                "HIDDEN_LIST": -1,
                "NUM_LAYERS": 50,
            },
            {
                "TYPE": "simple",
                "SIGMA": 2,
                "NUM_JOINTS": 17,
                "IMAGE_SIZE": [256, 192],
                "HEATMAP_SIZE": [64, 48],
            },
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def detect_2d_keypoints(self, frames: Frames) -> KeyPoints2D:

        model_input_size = (self.input_width, self.input_height)
        model_inputs, centers, scales = detect_bbox(frames, model_input_size)
        with torch.no_grad():
            model_outputs: list[RLEModelOutput] = [
                self.model(model_input)
                for model_input in tqdm(
                    model_inputs, desc="[Detecting 2d keypoints...]"
                )
            ]

        coords = list()
        for model_output, center, scale in zip(model_outputs, centers, scales):
            input_space_coords = (
                model_output.pred_jts + 0.5  # type: ignore
            ) * np.array([self.input_width, self.input_height])
            coords.append(
                transform_preds(input_space_coords[0], center, scale, model_input_size)
            )
        scores = np.stack(
            [model_output.maxvals[0, :, 0] for model_output in model_outputs]  # type: ignore
        )
        keypoints_2d = KeyPoints2D.from_coco(
            np.array(coords), scores, frames.width, frames.height
        )
        return keypoints_2d
