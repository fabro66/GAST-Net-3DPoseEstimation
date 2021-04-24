import numpy as np
import torch
from common.loss import mpjpe, p_mpjpe
from common.camera import *
from tools.utils import deterministic_random
from common.graph_utils import adj_mx_from_skeleton
from model.gast_net import *
from collections import OrderedDict
import os


def load_data(args):
    print("Loading dataset...")
    dataset_path = "data/data_3d_" + args.dataset + ".npz"
    if args.dataset == "h36m":
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path, args.keypoints)
    elif args.dataset.startswith('humaneva'):
        from common.humaneva_dataset import HumanEvaDataset
        dataset = HumanEvaDataset(dataset_path)
    else:
        raise KeyError("Invalid dataset")

    print("Preparing data...")
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if "positions" in anim:
                positions_3d = []
                for cam in anim["cameras"]:
                    pos_3d = world_to_camera(anim["positions"], R=cam["orientation"], t=cam["translation"])
                    pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim["positions_3d"] = positions_3d

    print("Loading 2D detections...")
    keypoints = np.load("data/data_2d_" + args.dataset + "_" + args.keypoints + ".npz", allow_pickle=True)
    keypoints_metadata = keypoints["metadata"].item()
    keypoints_metadata.update({'layout_name': 'h36m'})
    keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]

    if args.dataset.startswith('humaneva'):
        kps_left, kps_right = [2, 3, 4, 8, 9, 10], [5, 6, 7, 11, 12, 13]
    else:
        kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])

    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints["positions_2d"].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[
                subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if "positions_3d" not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]["positions_3d"])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]

                # HumanEva dataset detected from Mask-Rcnn with 17 keypoints
                # https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/keypoints.py
                # Transform the format of MSCOCO to the format of Human3.6M
                if args.dataset.startswith('humaneva'):
                    kps_15 = np.zeros((kps.shape[0], 15, kps.shape[2]), dtype=np.float32)
                    kps_15[:, 0] = (kps[:, 11] + kps[:, 12]) / 2
                    kps_15[:, 1] = (kps[:, 5] + kps[:, 6]) / 2
                    kps_15[:, 2] = kps[:, 5]
                    kps_15[:, 3] = kps[:, 7]
                    kps_15[:, 4] = kps[:, 9]
                    kps_15[:, 5] = kps[:, 6]
                    kps_15[:, 6] = kps[:, 8]
                    kps_15[:, 7] = kps[:, 10]
                    kps_15[:, 8] = kps[:, 11]
                    kps_15[:, 9] = kps[:, 13]
                    kps_15[:, 10] = kps[:, 15]
                    kps_15[:, 11] = kps[:, 12]
                    kps_15[:, 12] = kps[:, 14]
                    kps_15[:, 13] = kps[:, 16]
                    kps_15[:, 14] = kps[:, 0]

                    kps_15[..., :2] = normalize_screen_coordinates(kps_15[..., :2], w=cam["res_w"], h=cam["res_h"])
                    keypoints[subject][action][cam_idx] = kps_15

                else:
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam["res_w"], h=cam["res_h"])
                    keypoints[subject][action][cam_idx] = kps

    return keypoints, dataset, keypoints_metadata, kps_left, kps_right, joints_left, joints_right


def fetch(subjects, action_filter, dataset, keypoints, downsample=5, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


def create_model(args, dataset, poses_valid_2d):
    filter_widths = [int(x) for x in args.architecture.split(",")]
    adj = adj_mx_from_skeleton(dataset.skeleton())

    if not args.disable_optimizations and args.stride == 1:
        # Use optimized model for single-frame predictions
        model_pos_train = SpatioTemporalModelOptimized1f(adj, poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                                         dataset.skeleton().num_joints(), filter_widths=filter_widths,
                                                         causal=args.causal, dropout=args.dropout,
                                                         channels=args.channels)
    else:
        # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
        model_pos_train = SpatioTemporalModel(adj, poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                              dataset.skeleton().num_joints(), filter_widths=filter_widths,
                                              causal=args.causal, dropout=args.dropout, channels=args.channels)

    model_pos = SpatioTemporalModel(adj, poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                    dataset.skeleton().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                    channels=args.channels)

    receptive_field = model_pos.receptive_field()
    print("INFO: Receptive field: {} frames".format(receptive_field))
    pad = (receptive_field - 1) // 2  # padding on each side
    if args.causal:
        print("INFO: Using causal convolutions")
        causal_shift = pad
    else:
        causal_shift = 0

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print("INFO: Trainable parameter count: ", model_params)

    return model_pos_train, model_pos, pad, causal_shift


def load_weight(args, model_pos_train, model_pos):
    checkpoint = dict()
    if args.resume or args.evaluate:
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
        print("Loading checkpoint", chk_filename)
        checkpoint = torch.load(chk_filename)
        # print("This model was trained for {} epochs".format(checkpoint["epoch"]))
        model_pos_train.load_state_dict(checkpoint["model_pos"])
        model_pos.load_state_dict(checkpoint["model_pos"])

    return model_pos_train, model_pos, checkpoint


def train(model_pos_train, train_generator, optimizer):
    epoch_loss_3d_train = 0
    N = 0
    
    # Regular supervised scenario
    for _, batch_3d, batch_2d in train_generator.next_epoch():
        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()

        inputs_3d[:, :, 0] = 0

        optimizer.zero_grad()

        # Predict 3D poses
        predicted_3d_pos = model_pos_train(inputs_2d)
        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

        epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
        N += inputs_3d.shape[0] * inputs_3d.shape[1]

        loss_total = loss_3d_pos
        loss_total.backward()

        optimizer.step()

    epoch_losses_eva = epoch_loss_3d_train / N
    
    return epoch_losses_eva


def eval(model_train_dict, model_pos, test_generator, train_generator_eval):
    N = 0
    epoch_loss_3d_valid = 0
    epoch_loss_3d_train_eval = 0

    with torch.no_grad():
        model_pos.load_state_dict(model_train_dict)
        model_pos.eval()

        # Evaluate on test set
        for cam, batch, batch_2d in test_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            inputs_3d[:, :, 0] = 0

            # Predict 3D poses
            predicted_3d_pos = model_pos(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

        losses_3d_valid_ave = epoch_loss_3d_valid / N

        # Evaluate on training set, this time in evaluation mode
        N = 0
        for cam, batch, batch_2d in train_generator_eval.next_epoch():
            if batch_2d.shape[1] == 0:
                # This happens only when downsampling the dataset
                continue

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            inputs_3d[:, :, 0] = 0

            # Compute 3D poses
            predicted_3d_pos = model_pos(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

        losses_3d_train_eval_ave = epoch_loss_3d_train_eval / N

    return losses_3d_valid_ave, losses_3d_train_eval_ave


def evaluate(test_generator, model_pos, joints_left, joints_right, action=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
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

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()

            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000

    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')

    return e1, e2
