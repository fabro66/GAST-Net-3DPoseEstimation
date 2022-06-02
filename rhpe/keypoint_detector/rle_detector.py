from pathlib import Path
from typing import Protocol, Sequence
import numpy as np

import torch
from rhpe.core import Frames, KeyPointDetector2D, KeyPoints2D
from torch import nn
from rlepose.models import builder
from rlepose.utils.transforms import CropTransformer, im_to_torch, heatmap_to_coord


class RLEModelOutput(Protocol):
    pred_jts: Sequence[torch.Tensor]
    pred_scores: Sequence[torch.Tensor]


class RLEKeyPointDetector2D(KeyPointDetector2D):
    def __init__(
        self,
        device: str = "cuda:0",
    ):
        # Only support this input size for now.
        self.input_height, self.input_width = 256, 192
        self.device = device
        self.model = self.load_model(device)

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
    ) -> np.ndarray:
        """post process model output to create pose coordinates in original image space

        Args:
            model_output (RLEModelOutput): raw output of model
            transformer (CropTransformer): croptransformer

        Returns:
            np.ndarray: (F, J, 2) coordinates in original image space
        """
        num_frames = len(model_output.pred_jts)
        pred_jts = torch.stack(
            [model_output.pred_jts[idx] for idx in range(num_frames)]
        )
        pred_scores = torch.stack(
            [model_output.pred_scores[idx] for idx in range(num_frames)]
        )
        bbox = np.array([0, 0, self.input_width, self.input_height])
        coords: np.ndarray
        coords, _scores = heatmap_to_coord(
            pred_jts,
            pred_scores,
            (self.input_height // 4, self.input_width // 4),
            bbox,
            False,
        )  # F, J, 2
        coords_original = np.stack(
            [transformer.inverse_position(frame_coords) for frame_coords in coords]
        )
        return coords_original

    def detect_2d_keypoints(self, frames: Frames) -> KeyPoints2D:
        original_size = (frames.width, frames.height)  # x, y not y, x
        input_size = (self.input_width, self.input_height)  # x, y not y, x
        transformer = CropTransformer(original_size, input_size)
        model_input = self.preprocess_frames(frames, transformer)
        model_output = self.model(model_input)
        coords_original = self.postprocess_frames(model_output, transformer)
        return KeyPoints2D.from_coco(coords_original, frames.width, frames.height)
