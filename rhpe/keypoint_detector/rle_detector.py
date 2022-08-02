from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
from lib.pose.hrnet.lib.utils.transforms import transform_preds
from lib.pose.hrnet.pose_estimation.gen_kpts import detect_bbox
from rhpe.core import Frames, KeyPointDetector, KeyPoints2D
from rhpe.util.transform import CropTransformer
from rlepose.models import builder
from rlepose.utils.transforms import heatmap_to_coord, im_to_torch
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
        # original_size = (frames.width, frames.height)  # x, y not y, x
        # input_size = (self.input_width, self.input_height)  # x, y not y, x
        # transformer = CropTransformer(original_size, input_size)
        # model_input = self.preprocess_frames(frames, transformer)
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
            bbox = np.array([0, 0, self.input_width, self.input_height])
            input_space_coords = (model_output.pred_jts + 0.5) * np.array(
                [self.input_width, self.input_height]
            )
            coords.append(
                transform_preds(input_space_coords[0], center, scale, model_input_size)
            )
        scores = np.stack(
            [model_output.maxvals[0, :, 0] for model_output in model_outputs]
        )
        keypoints_2d = KeyPoints2D.from_coco(
            np.array(coords), scores, frames.width, frames.height
        )
        return keypoints_2d
