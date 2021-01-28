'''
Project: https://github.com/fabro66/GAST-Net-3DPoseEstimation
'''
import numpy as np


h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]

scores_h36m_toe_oeder = [1, 2, 3, 5, 6, 7, 11, 13, 14, 15, 16, 17, 18]
kpts_h36m_toe_order = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
scores_coco_order = [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]

h36m_mpii_order = [3, 2, 1, 4, 5, 6, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13]
mpii_order = [i for i in range(16)]
lr_hip_shouler = [2, 3, 12, 13]


def coco_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    keypoints_h36m[:, 7, 0] += 2*(keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])*2/3

    # half body: the joint of ankle and knee equal to hip
    # keypoints_h36m[:, [2, 3]] = keypoints_h36m[:, [1, 1]]
    # keypoints_h36m[:, [5, 6]] = keypoints_h36m[:, [4, 4]]

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    return keypoints_h36m, valid_frames


def mpii_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros((temporal, 17, 2), dtype=np.float32)
    keypoints_h36m[:, h36m_mpii_order] = keypoints
    # keypoints_h36m[:, 7] = np.mean(keypoints[:, 6:8], axis=1, dtype=np.float32)
    keypoints_h36m[:, 7] = np.mean(keypoints[:, lr_hip_shouler], axis=1, dtype=np.float32)

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    return keypoints_h36m, valid_frames


def coco_h36m_toe_format(keypoints):
    assert len(keypoints.shape) == 3
    temporal = keypoints.shape[0]

    new_kpts = np.zeros((temporal, 19, 2), dtype=np.float32)

    # convert body+foot keypoints
    coco_body_kpts = keypoints[:, :17].copy()
    h36m_body_kpts, _ = coco_h36m(coco_body_kpts)
    new_kpts[:, kpts_h36m_toe_order] = h36m_body_kpts
    new_kpts[:, 4] = np.mean(keypoints[:, [20, 21]], axis=1, dtype=np.float32)
    new_kpts[:, 8] = np.mean(keypoints[:, [17, 18]], axis=1, dtype=np.float32)

    valid_frames = np.where(np.sum(new_kpts.reshape(-1, 38), axis=-1) != 0)[0]

    return new_kpts,  valid_frames
