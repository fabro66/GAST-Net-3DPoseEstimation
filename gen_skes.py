import torch
import sys
import os.path as osp
import os
import argparse
import cv2
import time
import h5py
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.realpath(__file__)))
from tools.utils import get_path
from model.gast_net import SpatioTemporalModel, SpatioTemporalModelOptimized1f
# from imp_model.gast_net import SpatioTemporalModelOptimized1f
from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton
from common.generators import *
from tools.vis_h36m import render_animation
from tools.preprocess import load_kpts_json, h36m_coco_format, revise_kpts, revise_skes
from tools.inference import gen_pose
from tools.vis_kpts import plot_keypoint

cur_dir, chk_root, data_root, lib_root, output_root = get_path(__file__)
model_dir = chk_root + 'gastnet/'
sys.path.insert(1, lib_root)
from lib.pose import gen_video_kpts as hrnet_pose
sys.path.pop(1)
sys.path.pop(0)


skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
adj = adj_mx_from_skeleton(skeleton)

joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}
width, height = (1920, 1080)


def load_model_realtime(rf=81):
    if rf == 27:
        chk = model_dir + '27_frame_model_causal.bin'
        filters_width = [3, 3, 3]
        channels = 128
    elif rf == 81:
        chk = model_dir + '81_frame_model_causal.bin'
        filters_width = [3, 3, 3, 3]
        channels = 64
    else:
        raise ValueError('Only support 27 and 81 receptive field models for inference!')

    print('Loading GAST-Net ...')
    model_pos = SpatioTemporalModelOptimized1f(adj, 17, 2, 17, filter_widths=filters_width, causal=True,
                                               channels=channels, dropout=0.25)

    # Loading pre-trained model
    checkpoint = torch.load(chk)
    model_pos.load_state_dict(checkpoint['model_pos'])

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    model_pos.eval()

    print('GAST-Net successfully loaded')

    return model_pos


def load_model_layer(rf=27):
    if rf == 27:
        chk = model_dir + '27_frame_model.bin'
        filters_width = [3, 3, 3]
        channels = 128
    elif rf == 81:
        chk = model_dir + '81_frame_model.bin'
        filters_width = [3, 3, 3, 3]
        channels = 64
    else:
        raise ValueError('Only support 27 and 81 receptive field models for inference!')

    print('Loading GAST-Net ...')
    model_pos = SpatioTemporalModel(adj, 17, 2, 17, filter_widths=filters_width, channels=channels, dropout=0.05)

    # Loading pre-trained model
    checkpoint = torch.load(chk)
    model_pos.load_state_dict(checkpoint['model_pos'])

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    model_pos = model_pos.eval()

    print('GAST-Net successfully loaded')

    return model_pos


def generate_skeletons(video='', rf=27, output_animation=False, num_person=1, ab_dis=False):
    """
    :param video: The input video name. The video is placed in the Data folder
    :param output_npz: The output file. If the output_npz is set to true, the output_animation must be false
    :param rf: receptive fields
    :param output_animation: Generating animation video
    :param num_person: The number of 3D poses generated in the video. 1 or 2
    :param ab_dis: Whether the 3D pose generates the absolute distance of the plane (x, y)
    """

    # video = data_root + video
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    keypoints, scores = hrnet_pose(video, det_dim=416, num_peroson=num_person, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    num_person = len(re_kpts)

    # Loading 3D pose model
    model_pos = load_model_layer(rf)

    print('Generating 3D human pose ...')
    # pre-process keypoints

    pad = (rf - 1) // 2  # Padding on each side
    causal_shift = 0

    # Generating 3D poses
    prediction = gen_pose(re_kpts, valid_frames, width, height, model_pos, pad, causal_shift)

    # Adding absolute distance to 3D poses and rebase the height
    if num_person == 2:
        prediction = revise_skes(prediction, re_kpts, valid_frames)
    elif ab_dis:
        prediction[0][:, :, 2] -= np.expand_dims(np.amin(prediction[0][:, :, 2], axis=1), axis=1).repeat([17], axis=1)
    else:
        prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])

    # If output two 3D human poses, put them in the same 3D coordinate system
    same_coord = False
    if num_person == 2:
        same_coord = True

    anim_output = {}
    for i, anim_prediction in enumerate(prediction):
        anim_output.update({'Reconstruction %d' % (i+1): anim_prediction})

    if output_animation:
        viz_output = './output/' + 'animation_' + video.split('/')[-1].split('.')[0] + '.mp4'
        print('Generating animation ...')
        # re_kpts: (M, T, N, 2) --> (T, M, N, 2)
        re_kpts = re_kpts.transpose(1, 0, 2, 3)
        render_animation(re_kpts, keypoints_metadata, anim_output, skeleton, 25, 30000, np.array(70., dtype=np.float32),
                         viz_output, input_video_path=video, viewport=(width, height), com_reconstrcution=same_coord)
    else:
        print('Saving 3D reconstruction...')
        output_npz = './output/' + video.split('/')[-1].split('.')[0] + '.npz'
        np.savez_compressed(output_npz, reconstruction=prediction)
        print('Completing saving...')


def arg_parse():
    """
    Parse arguments for the skeleton module
    """
    parser = argparse.ArgumentParser('Generating skeleton demo.')
    parser.add_argument('-rf', '--receptive-field', type=int, default=81, help='number of receptive fields')
    parser.add_argument('-v', '--video', type=str, default='baseball.mp4', help='input video')
    parser.add_argument('-a', '--animation', action='store_true', help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1, help='number of estimated human poses. [1, 2]')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()
    video_path = data_root + 'video/' + args.video
    generate_skeletons(video=video_path, output_animation=args.animation, num_person=args.num_person)
