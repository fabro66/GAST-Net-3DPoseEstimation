import os.path as osp
import sys

import torch
from model.gast_net import SpatioTemporalModel
from rhpe.core import Frames, KeyPointDetector, KeyPoints2D

sys.path.insert(0, osp.dirname(osp.realpath(__file__)))
from common.generators import *
from common.graph_utils import adj_mx_from_skeleton
# from imp_model.gast_net import SpatioTemporalModelOptimized1f
from common.skeleton import Skeleton
from tools.utils import get_path
from torch import nn

cur_dir, chk_root, data_root, lib_root, output_root = get_path(__file__)
model_dir = chk_root + "gastnet/"
sys.path.insert(1, lib_root)
from lib.pose import gen_video_kpts_with_frames as hrnet_pose  # type: ignore

sys.path.pop(1)
sys.path.pop(0)


skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
)
adj = adj_mx_from_skeleton(skeleton)

joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
keypoints_metadata = {
    "keypoints_symmetry": (joints_left, joints_right),
    "layout_name": "Human3.6M",
    "num_joints": 17,
}


class HRNetDetector(KeyPointDetector):
    def __init__(
        self,
        device: torch.device | None = torch.device("cuda:0"),
    ):
        # Only support this input size for now.
        self.device = device

    def detect_2d_keypoints(self, frames: Frames) -> KeyPoints2D:
        keypoints, scores = hrnet_pose(
            frames, det_dim=416, num_peroson=1, gen_output=True
        )
        return KeyPoints2D.from_coco(
            keypoints[0], scores[0], frames.width, frames.height
        )


def load_model_layer(rf: int = 27, device: torch.device | None = None) -> nn.Module:
    if rf == 27:
        chk = model_dir + "27_frame_model.bin"
        filters_width = [3, 3, 3]
        channels = 128
    elif rf == 81:
        chk = model_dir + "81_frame_model.bin"
        filters_width = [3, 3, 3, 3]
        channels = 64
    else:
        raise ValueError("Only support 27 and 81 receptive field models for inference!")

    print("Loading GAST-Net ...")
    model_pos = SpatioTemporalModel(
        adj, 17, 2, 17, filter_widths=filters_width, channels=channels, dropout=0.05
    )

    # Loading pre-trained model
    checkpoint = torch.load(chk)
    model_pos.load_state_dict(checkpoint["model_pos"])

    if torch.cuda.is_available():
        model_pos = model_pos.cuda(device)
    model_pos = model_pos.eval()

    print("GAST-Net successfully loaded")

    return model_pos
