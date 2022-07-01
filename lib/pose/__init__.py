import os.path as osp
import sys

sys.path.insert(
    1, osp.join(osp.dirname(osp.realpath(__file__)), "hrnet/pose_estimation")
)
sys.path.insert(2, osp.join(osp.dirname(osp.realpath(__file__)), "hrnet/lib"))
from gen_kpts import (
    gen_img_kpts,
    gen_video_kpts,
    gen_video_kpts_with_frames,
    load_default_model,
)

sys.path.pop(2)

sys.path.insert(2, osp.join(osp.dirname(osp.realpath(__file__)), "hrnet/lib/utils"))
from utilitys import PreProcess, box_to_center_scale, load_json, plot_keypoint, write

sys.path.pop(1)
sys.path.pop(2)
