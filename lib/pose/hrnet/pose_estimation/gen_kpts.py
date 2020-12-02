from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2

import _init_paths
from _init_paths import get_path
from utils.utilitys import plot_keypoint, PreProcess, write, load_json
from config import cfg, update_config
from utils.transforms import *
from utils.inference import get_final_preds
import models
sys.path.pop(0)

pre_dir, cur_dir, chk_root, data_root, lib_root, output_root = get_path(__file__)
cfg_dir = pre_dir + '/experiments/coco/hrnet/'
model_dir = chk_root + 'hrnet/pose_coco/'

# Loading human detector model
sys.path.insert(0, lib_root)
from detector import load_model as yolo_model
from detector import yolo_human_det as yolo_det
from track.sort import Sort
sys.path.pop(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.70,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    args = parser.parse_args()

    return args


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    print('Loading HRNet model ...')
    # lib/models/pose_hrnet.py:get_pose_net
    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print('HRNet network successfully loaded')
    return model


def load_default_model():
    args = parse_args()
    reset_config(args)

    print('Loading HRNet model ...')
    # lib/models/pose_hrnet.py:get_pose_net
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(cfg.OUTPUT_DIR)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print('HRNet network successfully loaded')
    return model


def gen_img_kpts(image, human_model, pose_model, human_sort, det_dim=416, num_peroson=2):
    """
    :param image: Input image matrix instead of image path
    :param human_model: The YOLOv3 model
    :param pose_model: The HRNet model
    :param human_sort: Input initialized sort tracker
    :param det_dim: The input dimension of YOLOv3. [160, 320, 416]
    :param num_peroson: The number of tracked people

    :return:
            kpts: (M, N, 2)
            scores: (M, N, 1)
            bboxs_track: (x1, y1, x2, y2, ID)
            human_sort: Updated human_sort
    """

    args = parse_args()
    reset_config(args)

    thred_score = args.thred_score

    bboxs, bbox_scores = yolo_det(image, human_model, reso=det_dim, confidence=thred_score)

    if bboxs is None or not bboxs.any():
        return None, None, None

    # Using Sort to track people
    # people_track: Num_bbox × [x1, y1, x2, y2, ID]
    people_track = human_sort.update(bboxs)

    # Track the first two people in the video and remove the ID
    if people_track.shape[0] == 1:
        bboxs_track = people_track[-1].reshape(1, 5)
    else:
        people_track_ = people_track[-num_peroson:].reshape(num_peroson, 5)
        bboxs_track = people_track_[::-1]

    with torch.no_grad():
        # bbox is coordinate location
        inputs, origin_img, center, scale = PreProcess(image, bboxs_track, cfg, num_peroson)
        inputs = inputs[:, [2, 1, 0]]

        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = pose_model(inputs)

        # compute coordinate
        preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17, 1), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt
        for i, score in enumerate(maxvals):
            scores[i] = score

    human_indexes = []
    for i in range(len(bboxs_track)):
        human_indexes.append(bboxs_track[i, -1])

    return kpts, scores, human_indexes


def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False):
    # Updating configuration
    args = parse_args()
    reset_config(args)

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), 'Cannot capture source'

    # Loading detector and pose model, initialize sort for track
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort()

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # video_length = 1000

    # collect keypoints coordinate
    print('Generating 2D pose ...')

    kpts_result = []
    scores_result = []
    for i in tqdm(range(video_length)):
        ret, frame = cap.read()
        if not ret:
            continue
        # start = time.time()
        try:
            bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)

            if bboxs is None or not bboxs.any():
                print('No person detected!')
                # print('FPS of the video is {:5.2f}'.format(1 / (time.time() - start)))
                continue

            # Using Sort to track people
            people_track = people_sort.update(bboxs)

            # Track the first two people in the video and remove the ID
            if people_track.shape[0] == 1:
                people_track_ = people_track[-1, :-1].reshape(1, 4)
            elif people_track.shape[0] >= 2:
                people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
                people_track_ = people_track_[::-1]
            else:
                continue

            track_bboxs = []
            for bbox in people_track_:
                bbox = [round(i, 2) for i in list(bbox)]
                track_bboxs.append(bbox)

        except Exception as e:
            print(e)
            exit(0)
            continue

        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)
            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = pose_model(inputs)

            # compute coordinate
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        if gen_output:
            kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
            scores = np.zeros((num_peroson, 17), dtype=np.float32)
            for i, kpt in enumerate(preds):
                kpts[i] = kpt

            for i, score in enumerate(maxvals):
                scores[i] = score.squeeze()

            kpts_result.append(kpts)
            scores_result.append(scores)

        else:
            index_bboxs = [bbox + [i] for i, bbox in enumerate(track_bboxs)]
            list(map(lambda x: write(x, frame), index_bboxs))
            plot_keypoint(frame, preds, maxvals, 0.3)

            # print('FPS of the video is {:5.2f}'.format(1 / (time.time() - start)))
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    if gen_output:
        keypoints = np.array(kpts_result)
        scores = np.array(scores_result)

        keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
        scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)
        return keypoints, scores


def generate_ntu_kpts_json(video_path, kpts_file):
    args = parse_args()
    reset_config(args)

    # Loading detector and pose model, initialize sort for track
    human_model = yolo_model()
    pose_model = model_load(cfg)
    people_sort = Sort()

    with torch.no_grad():
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # collect keypoints information
        kpts_info = dict()
        data = []

        for i in tqdm(range(video_length)):
            frame_info = {'frame_index': i + 1}

            ret, frame = cap.read()
            try:
                bboxs, scores = yolo_det(frame, human_model, confidence=args.thred_score)

                if bboxs is None or not bboxs.any():
                    print('No person detected!')
                    continue
                # Using Sort to track people
                people_track = people_sort.update(bboxs)

                # Track the first two people in the video and remove the ID
                if people_track.shape[0] == 1:
                    people_track_ = people_track[-1, :-1].reshape(1, 4)
                elif people_track.shape[0] >= 2:
                    people_track_ = people_track[-2:, :-1].reshape(2, 4)
                    people_track_ = people_track_[::-1]
                else:
                    skeleton = {'skeleton': [{'pose': [], 'score': [], 'bbox': []}]}
                    frame_info.update(skeleton)
                    data.append(frame_info)

                    continue

                track_bboxs = []
                for bbox in people_track_:
                    bbox = [round(i, 3) for i in list(bbox)]
                    track_bboxs.append(bbox)

            except Exception as e:
                print(e)
                continue

            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, bboxs, cfg, args.num_person)
            inputs = inputs[:, [2, 1, 0]]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = pose_model(inputs.cuda())
            # compute coordinate
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center),
                                             np.asarray(scale))

            skeleton = []
            for num, bbox in enumerate(track_bboxs):
                pose = preds[num].tolist()
                score = maxvals[num].tolist()
                pose = round_list(pose)
                score = round_list(score)

                one_skeleton = {'pose': pose,
                                'score': score,
                                'bbox': bbox}
                skeleton.append(one_skeleton)

            frame_info.update({'skeleton': skeleton})
            data.append(frame_info)

        kpts_info.update({'data': data})
        with open(kpts_file, 'w') as fw:
            json.dump(kpts_info, fw)
    print('Finishing!')


def round_list(input_list, decimals=3):
    dim = len(input_list)

    for i in range(dim):
        for j in range(len(input_list[i])):
            input_list[i][j] = round(input_list[i][j], decimals)

    return input_list