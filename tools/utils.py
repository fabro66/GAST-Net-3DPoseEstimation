import torch
import numpy as np
import hashlib
import cv2
import os.path as osp


spple_keypoints = [10, 8, 0, 7]
h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
joint_pairs = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
               (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
colors_kps = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [50, 205, 50], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255]]


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        result = result.numpy()
        return result
    else:
        return result


def deterministic_random(min_value, max_value, data):
    """
        Encrypted, in order to generate the same size each time
    """

    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder="litter", signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value) + min_value)


def resize_img(frame, max_length=640):
    H, W = frame.shape[:2]
    if max(W, H) > max_length:
        if W > H:
            W_resize = max_length
            H_resize = int(H * max_length / W)
        else:
            H_resize = max_length
            W_resize = int(W * max_length / H)
        frame = cv2.resize(frame, (W_resize, H_resize), interpolation=cv2.INTER_AREA)
        return frame, W_resize, H_resize

    else:
        return frame, W, H


def draw_2Dimg(img, kpts, scores, display=None):
    # kpts : (M, 17, 2)  scores: (M, 17)
    im = img.copy()
    for kpt, score in zip(kpts, scores):
        for i, item in enumerate(kpt):
            score_val = score[i]
            if score_val > 0.3:
                x, y = int(item[0]), int(item[1])
                cv2.circle(im, (x, y), 4, (255, 255, 255), 1)
        for pair, color in zip(joint_pairs, colors_kps):
            j, j_parent = pair
            pt1 = (int(kpt[j][0]), int(kpt[j][1]))
            pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
            cv2.line(im, pt1, pt2, color, 2)

    if display:
        cv2.imshow('frame', im)
        cv2.waitKey(1)
    return im


def get_path(cur_file):
    project_root = osp.dirname(osp.realpath(cur_file))
    chk_root = osp.join(project_root, 'checkpoint/')
    data_root = osp.join(project_root, 'data/')
    lib_root = osp.join(project_root, 'lib/')
    output_root = osp.join(project_root, 'output/')

    return project_root, chk_root, data_root, lib_root, output_root


def coco_h36m_frame(keypoints):
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((4, 2), dtype=np.float32)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[0, 0] = np.mean(keypoints[1:5, 0], axis=0, dtype=np.float32)
    htps_keypoints[0, 1] = np.sum(keypoints[1:3, 1], axis=0, dtype=np.float32) - keypoints[0, 1]
    htps_keypoints[1, :] = np.mean(keypoints[5:7, :], axis=0, dtype=np.float32)
    htps_keypoints[1, :] += (keypoints[0, :] - htps_keypoints[1, :]) / 3

    htps_keypoints[2, :] = np.mean(keypoints[11:13, :], axis=0, dtype=np.float32)
    htps_keypoints[3, :] = np.mean(keypoints[[5, 6, 11, 12], :], axis=0, dtype=np.float32)

    keypoints_h36m[spple_keypoints, :] = htps_keypoints
    keypoints_h36m[h36m_coco_order, :] = keypoints[coco_order, :]

    keypoints_h36m[9, :] -= (keypoints_h36m[9, :] - np.mean(keypoints[5:7, :], axis=0, dtype=np.float32)) / 4
    keypoints_h36m[7, 0] += 0.3 * (keypoints_h36m[7, 0] - np.mean(keypoints_h36m[[0, 8], 0], axis=0, dtype=np.float32))
    keypoints_h36m[8, 1] -= (np.mean(keypoints[1:3, 1], axis=0, dtype=np.float32) - keypoints[0, 1]) * 2 / 3

    return keypoints_h36m


def h36m_coco_kpts(keypoints, scores):
    # keypoints: (M, N, C)  scores:(M, N, 1)
    assert len(keypoints.shape) == 3 and len(scores.shape) == 3
    scores.squeeze(axis=2)

    h36m_kpts = []
    h36m_scores = []
    for i in range(keypoints.shape[0]):
        kpts = keypoints[i]
        score = scores[i]

        new_score = np.zeros_like(score, dtype=np.float32)

        if np.sum(kpts) != 0.:
            new_score[h36m_coco_order] = score[coco_order]
            new_score[0] = np.mean(score[[11, 12]], axis=0, dtype=np.float32)
            new_score[8] = np.mean(score[[5, 6]], axis=0, dtype=np.float32)
            new_score[7] = np.mean(new_score[[0, 8]], axis=0, dtype=np.float32)
            new_score[10] = np.mean(score[[1, 2, 3, 4]], axis=0, dtype=np.float32)

            h36m_scores.append(new_score)

            kpts = coco_h36m_frame(kpts)
            less_threshold_joints = np.where(new_score < 0.3)[0]
            intersect = [i for i in [2, 3, 5, 6] if i in less_threshold_joints]

            if [2, 3, 5, 6] == intersect:
                kpts[[2, 3, 5, 6]] = kpts[[1, 1, 4, 4]]
            elif [2, 3, 6] == intersect:
                kpts[[2, 3, 6]] = kpts[[1, 1, 5]]
            elif [3, 5, 6] == intersect:
                kpts[[3, 5, 6]] = kpts[[2, 4, 4]]
            elif [3, 6] == intersect:
                kpts[[3, 6]] = kpts[[2, 5]]
            elif [3] == intersect:
                kpts[3] = kpts[2]
            elif [6] == intersect:
                kpts[6] = kpts[5]

            h36m_kpts.append(kpts)

    return h36m_kpts, h36m_scores
