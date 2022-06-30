import torch
from torch import nn


def get_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device
