from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from rhpe.core import Frames, KeyPointDetector, KeyPoints2D
from rhpe.keypoints_2d.h36m_keypoints_2d import H36MKeyPoints2D, RevisedH36MKeyPoints2D
from rhpe.util.transform import CropTransformer
from rlepose.models import builder
from rlepose.utils.transforms import heatmap_to_coord, im_to_torch
from torch import nn


class RLEModelOutput(Protocol):
    pred_jts: Sequence[torch.Tensor]
    maxvals: Sequence[torch.Tensor]


class RLEKeyPointDetector2D(KeyPointDetector):
    def __init__(
        self,
        revise: bool = True,
        device: str = "cuda:0",
    ):
        # Only support this input size for now.
        self.input_height, self.input_width = 256, 192
        self.device = device
        self.model = self.load_model(device)
        self.revise = revise

    def load_model(self, device: str) -> nn.Module:
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
        model_path = Path().joinpath("pretrained_model", "coco-laplace-rle.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def preprocess_frames(
        self, frames: Frames, transformer: CropTransformer
    ) -> torch.Tensor:
        """preprocess frames and create input tensor for 2d model.

        Args:
            frames (Frames): original frames to be processed

        Returns:
            torch.Tensor: F, 3, H, W
        """

        cropped_frames = torch.stack(
            [im_to_torch(transformer.transform_image(frame)) for frame in frames.numpy]
        )  # F, 3, H', W'
        # subtract constants following conventions.
        cropped_frames[0].add_(-0.406)
        cropped_frames[1].add_(-0.457)
        cropped_frames[2].add_(-0.480)
        return cropped_frames

    def postprocess_frames(
        self, model_output: RLEModelOutput, transformer: CropTransformer
    ) -> tuple[np.ndarray, np.ndarray]:
        """post process model output to create pose coordinates in original image space

        Args:
            model_output (RLEModelOutput): raw output of model
            transformer (CropTransformer): croptransformer

        Returns:
            tuple(np.ndarray, np.ndarray):
                (F, J, 2) coordinates in original image space,
                (F, J) confidence scores
        """
        num_frames = len(model_output.pred_jts)
        pred_jts = torch.stack(
            [model_output.pred_jts[idx] for idx in range(num_frames)]
        )
        pred_scores = torch.stack(
            [model_output.maxvals[idx] for idx in range(num_frames)]
        )
        bbox = np.array([0, 0, self.input_width, self.input_height])
        coords: np.ndarray
        coords, scores = heatmap_to_coord(
            pred_jts,
            pred_scores,
            (self.input_height // 4, self.input_width // 4),
            bbox,
            False,
        )  # F, J, 2
        coords_original = np.stack(
            [transformer.inverse_position(frame_coords) for frame_coords in coords]
        )
        scores = scores[:, :, 0]
        return coords_original, scores

    def detect_2d_keypoints(self, frames: Frames) -> KeyPoints2D:
        original_size = (frames.width, frames.height)  # x, y not y, x
        input_size = (self.input_width, self.input_height)  # x, y not y, x
        transformer = CropTransformer(original_size, input_size)
        model_input = self.preprocess_frames(frames, transformer)
        with torch.no_grad():
            model_output = self.model(model_input)
        coords_original, scores = self.postprocess_frames(model_output, transformer)
        keypoints_2d = H36MKeyPoints2D.from_coco(
            coords_original, scores, frames.width, frames.height
        )
        if self.revise:
            keypoints_2d = RevisedH36MKeyPoints2D.from_h36m(keypoints_2d)
        return keypoints_2d
