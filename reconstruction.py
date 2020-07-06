import torch
import torch.nn as nn
import torch
import numpy as np
import argparse

from tool.mpii_coco_h36m import coco_h36m, mpii_h36m
from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton
from common.camera import normalize_screen_coordinates, camera_to_world
from common.generators import *
from model.gast_net import *
from tool.visualization import render_animation


h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
adj = adj_mx_from_skeleton(h36m_skeleton)
joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}

mpii_metadata = {
    'layout_name': 'mpii',
    'num_joints': 16,
    'keypoints_symmetry': [
        [3, 4, 5, 13, 14, 15],
        [0, 1, 2, 10, 11, 12],
    ]
}

coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}

h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16],
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-c', '--checkpoint', type=str, metavar='NAME', help='The file path of model weight')
    parser.add_argument('-k', '--keypoints', type=str, metavar='NAME', help='The file path of 2D keypoints')
    parser.add_argument('-vi', '--video-path', type=str, metavar='NAME', help='The input video path')
    parser.add_argument('-vo', '--viz-output', type=str, metavar='NAME', help='The output path of animation')
    parser.add_argument('-kf', '--kpts-format', type=str, default='h36m', metavar='NAME', help='The format of 2D keypoints')

    return parser


def evaluate(test_generator, model_pos, return_predictions=False):
    
    with torch.no_grad():
        model_pos.eval()

        for _, batch, batch_2d in test_generator.next_epoch():

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


def reconstruction(chk_file, kps_file, viz_output, video_path=None, kpts_format='h36m'):
    """
    Generate 3D poses from 2D keypoints detected from video, and visualize it
        :param chk_file: The file path of model weight
        :param kps_file: The file path of 2D keypoints
        :param viz_output: The output path of animation
        :param video_path: The input video path
        :param kpts_format: The format of 2D keypoints, like MSCOCO, MPII, H36M. The default format is H36M
    """

    print('Loading 2D keypoints ...')
    data = np.load(kps_file, allow_pickle=True)

    # keypoints: (T, N, C)
    keypoints = data["keypoints"]
    img_h_w = data["img_h_w"]
    high, width = img_h_w[:2]

    # Transform the keypoints format from different dataset (MSCOCO, MPII) to h36m format
    if kpts_format == 'coco':
        keypoints = coco_h36m(keypoints)
    elif kpts_format == 'mpii':
        keypoints = mpii_h36m(keypoints)
    else:
        assert kpts_format == 'h36m'

    # normalize keypoints
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=width, h=high)

    model_pos = SpatioTemporalModel(adj, 17, 2, 17, filter_widths=[3, 3, 3], channels=128, dropout=0.05)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    # load trained model
    print('Loading checkpoint', chk_file)
    checkpoint = torch.load(chk_file, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])

    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    print('Reconstructing ...')
    input_keypoints = keypoints.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=True,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)

    prediction = camera_to_world(prediction, R=rot, t=0)

    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    print('Rendering ...')
    anim_output = {'Reconstruction': prediction}
    render_animation(keypoints, keypoints_metadata, anim_output, h36m_skeleton, 25, 3000,
                     np.array(70., dtype=np.float32), viz_output, limit=-1, downsample=1, size=5,
                     input_video_path=video_path, viewport=(width, high), input_video_skip=0)


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    # chk_file = '.../epoch_60.bin'
    # kps_file = '.../2d_keypoints.npz'
    # video_path = '.../sittingdown.mp4'
    # viz_output = '.../output_animation.mp4'

    reconstruction(args.checkpoint, args.keypoints, args.viz_output, args.video_path, args.kpts_format)
