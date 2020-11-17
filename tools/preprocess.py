import json
import numpy as np
from tools.mpii_coco_h36m import coco_h36m
import os


h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
num_person = 2
num_joints = 17
img_3d = 100.
ratio_2d_3d = 500.


def load_json(file_path):
    with open(file_path, 'r') as fr:
        video_info = json.load(fr)

    label = video_info['label']
    label_index = video_info['label_index']

    num_frames = video_info['data'][-1]['frame_index']
    keypoints = np.zeros((num_person, num_frames, num_joints, 2), dtype=np.float32)
    scores = np.zeros((num_person, num_frames, num_joints), dtype=np.float32)

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

    return keypoints, scores, label, label_index


def h36m_coco_format(keypoints, scores):
    assert len(keypoints.shape) == 4 and len(scores.shape) == 3

    h36m_kpts = []
    h36m_scores = []
    valid_frames = []

    for i in range(keypoints.shape[0]):
        kpts = keypoints[i]
        score = scores[i]

        new_score = np.zeros_like(score, dtype=np.float32)

        if np.sum(kpts) != 0.:
            kpts, valid_frame = coco_h36m(kpts)
            h36m_kpts.append(kpts)
            valid_frames.append(valid_frame)

            new_score[:, h36m_coco_order] = score[:, coco_order]
            new_score[:, 0] = np.mean(score[:, [11, 12]], axis=1, dtype=np.float32)
            new_score[:, 8] = np.mean(score[:, [5, 6]], axis=1, dtype=np.float32)
            new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
            new_score[:, 10] = np.mean(score[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

            h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)
    return h36m_kpts, h36m_scores, valid_frames


def revise_kpts(h36m_kpts, h36m_scores, valid_frames):

    new_h36m_kpts = np.zeros_like(h36m_kpts)
    for index, frames in enumerate(valid_frames):
        kpts = h36m_kpts[index, frames]
        score = h36m_scores[index, frames]

        # threshold_score = score > 0.3
        # if threshold_score.all():
        #     continue

        index_frame = np.where(np.sum(score < 0.3, axis=1) > 0)[0]

        for frame in index_frame:
            less_threshold_joints = np.where(score[frame] < 0.3)[0]

            intersect = [i for i in [2, 3, 5, 6] if i in less_threshold_joints]

            if [2, 3, 5, 6] == intersect:
                kpts[frame, [2, 3, 5, 6]] = kpts[frame, [1, 1, 4, 4]]
            elif [2, 3, 6] == intersect:
                kpts[frame, [2, 3, 6]] = kpts[frame, [1, 1, 5]]
            elif [3, 5, 6] == intersect:
                kpts[frame, [3, 5, 6]] = kpts[frame, [2, 4, 4]]
            elif [3, 6] == intersect:
                kpts[frame, [3, 6]] = kpts[frame, [2, 5]]
            elif [3] == intersect:
                kpts[frame, 3] = kpts[frame, 2]
            elif [6] == intersect:
                kpts[frame, 6] = kpts[frame, 5]
            else:
                continue

        new_h36m_kpts[index, frames] = kpts
    return new_h36m_kpts


def load_kpts_json(kpts_json):
    keypoints, scores, label, label_index = load_json(kpts_json)
    h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(h36m_kpts, h36m_scores, valid_frames)

    return re_kpts, valid_frames, scores, label, label_index


def revise_skes(prediction, re_kpts, valid_frames):
    new_prediction = np.zeros((*re_kpts.shape[:-1], 3), dtype=np.float32)
    for i, frames in enumerate(valid_frames):
        new_prediction[i, frames] = prediction[i]

        # The origin of (x, y) is in the upper right corner,
        # while the (x,y) coordinates in the image are in the upper left corner.
        distance = re_kpts[i, frames[1:], :, :2] - re_kpts[i, frames[:1], :, :2]
        distance = np.mean(distance[:, [1, 4, 11, 14]], axis=-2, keepdims=True)
        new_prediction[i, frames[1:], :, 0] -= distance[..., 0] / ratio_2d_3d
        new_prediction[i, frames[1:], :, 1] += distance[..., 1] / ratio_2d_3d

    # The origin of (x, y) is in the upper right corner,
    # while the (x,y) coordinates in the image are in the upper left corner.
    # Calculate the relative distance between two people
    if len(valid_frames) == 2:
        intersec_frames = [frame for frame in valid_frames[0] if frame in valid_frames[1]]
        absolute_distance = re_kpts[0, intersec_frames[:1], :, :2] - re_kpts[1, intersec_frames[:1], :, :2]
        absolute_distance = np.mean(absolute_distance[:, [1, 4, 11, 14]], axis=-2, keepdims=True) / 2.

        new_prediction[0, valid_frames[0], :, 0] -= absolute_distance[..., 0] / ratio_2d_3d
        new_prediction[0, valid_frames[0], :, 1] += absolute_distance[..., 1] / ratio_2d_3d

        new_prediction[1, valid_frames[1], :, 0] += absolute_distance[..., 0] / ratio_2d_3d
        new_prediction[1, valid_frames[1], :, 1] -= absolute_distance[..., 1] / ratio_2d_3d

    # Pre-processing the case where the movement of Z axis is relatively large, such as 'sitting down'
    # Remove the absolute distance
    # new_prediction[:, :, 1:] -= new_prediction[:, :, :1]
    # new_prediction[:, :, 0] = 0
    new_prediction[:, :, :, 2] -= np.amin(new_prediction[:, :, :, 2])

    return new_prediction


def revise_skes_real_time(prediction, re_kpts, width):
    ratio_2d_3d_width = ratio_2d_3d * (width / 1920)
    # prediction: (M, N, 3)
    new_prediction = np.zeros((len(prediction), 17, 3), dtype=np.float32)
    for i in range(len(prediction)):
        new_prediction[i] = prediction[i]

        initial_distance = re_kpts[i]
        initial_distance = np.mean(initial_distance[[1, 4, 11, 14], :], axis=0)
        new_prediction[i, :, 0] -= (initial_distance[0] - 3*width/5) / ratio_2d_3d_width
        new_prediction[i, :, 1] += (initial_distance[1] - width/5) / ratio_2d_3d_width

    new_prediction[:, :, 2] -= np.amin(new_prediction[:, :, 2])

    return new_prediction
