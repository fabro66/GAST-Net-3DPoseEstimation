import torch


def qort(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    """
    The quaternions provided in the code are from the camera coordinate to the world coordinate.
    Therefore, the quaternions from the world coordinate to the camera coordinate is the transpose of quaternions from
    the camera coordinates to the world coordinate.The precondition is that the quaternion is a unit quaternion.
    So the inverse of the quaternions is equal to the transposition of the quaternions.
    """
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)

