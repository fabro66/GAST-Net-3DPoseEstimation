import numpy as np
import torch

from tools.utils import wrap
from common.quaternion import qort, qinverse


def normalize_screen_coordinates(X, w, h):
    # Normalize so that x in [0, w] is mapped to [-1, 1]
    # y in [0, h] is mapped to [-h/w, h/w] to keep aspect ratio
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def normalize_minmax_coordinates(X: np.ndarray, w: int, h: int) -> np.ndarray:
    # Normalize so that [min, max] is mapped to [-1, 1], while preserving the aspect ratio
    assert len(X.shape) == 3
    assert X.shape[-1] == 2

    size = np.array([w, h], dtype=X.dtype)
    minimum = np.min(X, axis=(0, 1))  # (2,)
    maximum = np.max(X, axis=(0, 1))  # (2,)
    minimum = np.where(minimum < 0, minimum, 0)
    maximum = np.where(maximum > size, maximum, size)
    normalized = 2 * (X - minimum) / (maximum - minimum) - 1  # type: ignore
    # keep aspect ratio:
    value_range = maximum - minimum  # type: ignore
    normalized = normalized * np.array([1, value_range[1] / value_range[0]])
    return normalized


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R)  # Invert rotation
    return wrap(qort, np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


def camera_to_world(X, R, t):
    return wrap(qort, np.tile(R, (*X.shape[:-1], 1)), X) + t


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    # XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    XX = X[..., :2] / X[..., 2:]
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(
        k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape) - 1),
        dim=len(r2.shape) - 1,
        keepdim=True,
    )
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c
