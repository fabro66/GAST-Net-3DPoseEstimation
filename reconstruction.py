import torch
import torch.nn as nn
import torch
import numpy as np
import json
import cv2
import os
import argparse

from tools.mpii_coco_h36m import coco_h36m, mpii_h36m, coco_h36m_toe_format
from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton
from common.camera import normalize_screen_coordinates, camera_to_world
from common.generators import *
from model.gast_net import *
from tools.visualization import render_animation


# h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
#                          joints_left=[4, 5, 6, 11, 12, 13],
#                          joints_right=[1, 2, 3, 14, 15, 16])
# adj = adj_mx_from_skeleton(h36m_skeleton)
# body_joints_left, body_joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
# body_kps_left, body_kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
# keypoints_metadata = {'keypoints_symmetry': (body_joints_left, body_joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)


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
    parser.add_argument('-f', '--frames', type=int, default=27, metavar='NAME',
                        help='The number of receptive fields')
    parser.add_argument('-ca', '--causal', action='store_true',
                        help='Using real-time model with causal convolution')
    parser.add_argument('-w', '--weight', type=str, default='27_frame_model.bin', metavar='NAME',
                        help='The name of model weight')
    parser.add_argument('-n', '--num-joints', type=int, default=17, metavar='NAME',
                        help='The number of joints')
    parser.add_argument('-k', '--keypoints-file', type=str, default='./data/keypoints/baseball.json', metavar='NAME',
                        help='The path of keypoints file')
    parser.add_argument('-vi', '--video-path', type=str, default='./data/video/baseball.mp4', metavar='NAME',
                        help='The path of input video')
    parser.add_argument('-vo', '--viz-output', type=str, default='./output/baseball.mp4', metavar='NAME',
                        help='The path of output video')
    parser.add_argument('-kf', '--kpts-format', type=str, default='coco', metavar='NAME',
                        help='The format of 2D keypoints')

    return parser


def get_joints_info(num_joints):
    # Body+toe keypoints
    if num_joints == 19:
        joints_left = [5, 6, 7, 8, 13, 14, 15]
        joints_right = [1, 2, 3, 4, 16, 17, 18]

        h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 10, 16, 17],
                                 joints_left=[5, 6, 7, 8, 13, 14, 15],
                                 joints_right=[1, 2, 3, 4, 16, 17, 18])
    # Body keypoints
    else:
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                                 joints_left=[4, 5, 6, 11, 12, 13],
                                 joints_right=[1, 2, 3, 14, 15, 16])

    keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M',
                          'num_joints': num_joints}

    return joints_left, joints_right, h36m_skeleton, keypoints_metadata


def load_json(file_path, num_joints, num_person=2):
    with open(file_path, 'r') as fr:
        video_info = json.load(fr)

    # Loading whole-body keypoints including body(17)+hand(42)+foot(6)+facial(68) joints
    # 2D Whole-body human pose estimation paper: https://arxiv.org/abs/2007.11858
    if num_joints == 19:
        num_joints_revise = 133
    else:
        num_joints_revise = 17

    label = video_info['label']
    label_index = video_info['label_index']

    num_frames = video_info['data'][-1]['frame_index']
    keypoints = np.zeros((num_person, num_frames, num_joints_revise, 2), dtype=np.float32)
    scores = np.zeros((num_person, num_frames, num_joints_revise), dtype=np.float32)

    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']

        for index, skeleton_info in enumerate(frame_info['skeleton']):
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            bbox = skeleton_info['bbox']

            if len(bbox) == 0 or index+1 > num_person:
                continue

            pose = np.asarray(pose, dtype=np.float32)
            score = np.asarray(score, dtype=np.float32)
            score = score.reshape(-1)

            keypoints[index, frame_index-1] = pose
            scores[index, frame_index-1] = score

    if num_joints != num_joints_revise:
        # body(17) + foot(6) = 23
        return keypoints[:, :, :23], scores[:, :, :23], label, label_index
    else:
        return keypoints, scores, label, label_index


def evaluate(test_generator, model_pos, joints_left, joints_right, return_predictions=False):
    
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


def reconstruction(args):
    """
    Generate 3D poses from 2D keypoints detected from video, and visualize it
        :param chk_file: The file path of model weight
        :param kps_file: The file path of 2D keypoints
        :param viz_output: The output path of animation
        :param video_path: The input video path
        :param kpts_format: The format of 2D keypoints, like MSCOCO, MPII, H36M, OpenPose. The default format is H36M
    """

    # Getting joint information
    joints_left, joints_right, h36m_skeleton, keypoints_metadata = get_joints_info(args.num_joints)
    kps_left, kps_right = joints_left, joints_right
    adj = adj_mx_from_skeleton(h36m_skeleton)

    print('Loading 2D keypoints ...')
    keypoints, scores, _, _ = load_json(args.keypoints_file, args.num_joints)

    # Loading only one person's keypoints
    if len(keypoints.shape) == 4:
        keypoints = keypoints[0]
    assert len(keypoints.shape) == 3

    # Transform the keypoints format from different dataset (MSCOCO, MPII) to h36m format
    if args.kpts_format == 'coco':
        keypoints, valid_frames = coco_h36m(keypoints)
    elif args.kpts_format == 'mpii':
        keypoints, valid_frames = mpii_h36m(keypoints)
    elif args.kpts_format == 'openpose':
        # Convert 'Openpose' format to MSCOCO
        order_coco = [i for i in range(18) if i != 1]
        keypoints = keypoints[:, order_coco]
        keypoints, valid_frames = coco_h36m(keypoints)
    elif args.kpts_format == 'wholebody':
        keypoints, valid_frames = coco_h36m_toe_format(keypoints)
    else:
        valid_frames = np.where(np.sum(keypoints.reshape(-1, 34), axis=1) != 0)[0]
        assert args.kpts_format == 'h36m'

    # Get the width and height of video
    cap = cv2.VideoCapture(args.video_path)
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # normalize keypoints
    input_keypoints = normalize_screen_coordinates(keypoints[..., :2], w=width, h=height)

    if args.frames == 27:
        filter_widths = [3, 3, 3]
        channels = 128
    elif args.frames == 81:
        filter_widths = [3, 3, 3, 3]
        channels = 64
    else:
        filter_widths = [3, 3, 3, 3, 3]
        channels = 32

    model_pos = SpatioTemporalModel(adj, args.num_joints, 2, args.num_joints, filter_widths=filter_widths,
                                    channels=channels, dropout=0.05, causal=args.causal)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    # load pretrained model
    print('Loading checkpoint', args.weight)
    chk_file = os.path.join('./checkpoint/gastnet', args.weight)
    checkpoint = torch.load(chk_file, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])

    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side

    if args.causal:
        causal_shift = pad
    else:
        causal_shift = 0

    print('Reconstructing ...')
    gen = UnchunkedGenerator(None, None, [input_keypoints[valid_frames]],
                             pad=pad, causal_shift=causal_shift, augment=True,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos, joints_left, joints_right, return_predictions=True)
    prediction = camera_to_world(prediction, R=rot, t=0)

    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    prediction_new = np.zeros((*input_keypoints.shape[:-1], 3), dtype=np.float32)
    prediction_new[valid_frames] = prediction

    print('Rendering ...')
    anim_output = {'Reconstruction': prediction_new}
    render_animation(keypoints, keypoints_metadata, anim_output, h36m_skeleton, 25, 3000,
                     np.array(70., dtype=np.float32), args.viz_output, limit=-1, downsample=1, size=5,
                     input_video_path=args.video_path, viewport=(width, height), input_video_skip=0)


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    # chk_file = '.../epoch_60.bin'
    # kps_file = '.../2d_keypoints.npz'
    # video_path = '.../sittingdown.mp4'
    # viz_output = '.../output_animation.mp4'

    reconstruction(args)
