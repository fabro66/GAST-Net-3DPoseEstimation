import torch
from torchsummary import summary
import torch.nn as nn
from model.local_attention import LocalGraph
from model.global_attention import MultiGlobalGraph, SingleGlobalGraph


class GraphAttentionBlock(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout):
        super(GraphAttentionBlock, self).__init__()
        
        hid_dim = output_dim
        self.relu = nn.ReLU(inplace=True)

        self.local_graph_layer = LocalGraph(adj, input_dim, hid_dim, p_dropout)
        self.global_graph_layer = MultiGlobalGraph(adj, input_dim, input_dim//4, dropout=p_dropout)
        # self.global_graph_layer = SingleGlobalGraph(adj, input_dim, output_dim)

        self.cat_conv = nn.Conv2d(3*output_dim, 2*output_dim, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(2*output_dim, momentum=0.1)

    def forward(self, x):
        # x: (B, C, T, N) --> (B, T, N, C)
        x = x.permute(0, 2, 3, 1)
        residual = x
        x_ = self.local_graph_layer(x)
        y_ = self.global_graph_layer(x)
        x = torch.cat((residual, x_, y_), dim=-1)

        # x: (B, T, N, C) --> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.cat_bn(self.cat_conv(x)))
        return x


class SpatioTemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.init_bn = nn.BatchNorm2d(in_features, momentum=0.1)
        self.expand_bn = nn.BatchNorm2d(channels, momentum=0.1)
        self.shrink = nn.Conv2d(2**len(self.filter_widths)*channels, 3, 1, bias=False)

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        """
        X: (B, C, T, N)
            B: batchsize
            T: Temporal
            N: The number of keypoints
            C: The feature dimension of keypoints
        """

        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        # X: (B, T, N, C)
        x = self._forward_blocks(x)
        x = self.shrink(x)

        # x: (B, C, T, N) --> (B, T, N, C)
        x = x.permute(0, 2, 3, 1)

        return x


class SpatioTemporalModel(SpatioTemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=64, dense=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(adj, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv2d(in_features, channels, (filter_widths[0], 1), bias=False)
        nn.init.kaiming_normal_(self.expand_conv.weight)

        layers_conv = []
        layers_graph_conv = []
        layers_bn = []

        layers_graph_conv.append(GraphAttentionBlock(adj, channels, channels, p_dropout=dropout))

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(nn.Conv2d(2**i*channels, 2**i*channels, (filter_widths[i], 1) if not dense else (2*self.pad[-1]+1, 1),
                               dilation=(next_dilation, 1) if not dense else (1, 1), bias=False))
            layers_bn.append(nn.BatchNorm2d(2**i*channels, momentum=0.1))
            layers_conv.append(nn.Conv2d(2**i*channels, 2**i*channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm2d(2**i*channels, momentum=0.1))

            layers_graph_conv.append(GraphAttentionBlock(adj, 2**i*channels, 2**i*channels, p_dropout=dropout))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_graph_conv = nn.ModuleList(layers_graph_conv)

    def _forward_blocks(self, x):

        # x: (B, T, N, C) --> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.init_bn(x)
        x = self.relu(self.expand_bn(self.expand_conv(x)))
        x = self.layers_graph_conv[0](x)

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]

            # x: (B, C, T, N)
            x = self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x)))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

            x = self.layers_graph_conv[i + 1](x)
        return x


class SpatioTemporalModelOptimized1f(SpatioTemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=64):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(adj, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv2d(in_features, channels, (filter_widths[0], 1), stride=(filter_widths[0], 1), bias=False)
        nn.init.kaiming_normal_(self.expand_conv.weight)

        layers_conv = []
        layers_graph_conv = []
        layers_bn = []

        layers_graph_conv.append(GraphAttentionBlock(adj, channels, channels, p_dropout=dropout))

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_conv.append(nn.Conv2d(2**i*channels, 2**i*channels, (filter_widths[i], 1), stride=(filter_widths[i], 1), bias=False))
            layers_bn.append(nn.BatchNorm2d(2**i*channels, momentum=0.1))
            layers_conv.append(nn.Conv2d(2**i*channels, 2**i*channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm2d(2**i*channels, momentum=0.1))

            layers_graph_conv.append(GraphAttentionBlock(adj, 2**i*channels, 2**i*channels, p_dropout=dropout))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_graph_conv = nn.ModuleList(layers_graph_conv)

    def _forward_blocks(self, x):
        # x: (B, T, N, C) --> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.init_bn(x)
        x = self.relu(self.expand_bn(self.expand_conv(x)))
        x = self.layers_graph_conv[0](x)

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]

            # x: (B, C, T, N)
            x = self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x)))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

            x = self.layers_graph_conv[i+1](x)

        return x


if __name__ == "__main__":
    import torch
    import numpy as np
    import torchsummary
    from common.skeleton import Skeleton
    from common.graph_utils import adj_mx_from_skeleton

    h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                             joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                             joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

    humaneva_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
                                 joints_left=[2, 3, 4, 8, 9, 10],
                                 joints_right=[5, 6, 7, 11, 12, 13])

    adj = adj_mx_from_skeleton(h36m_skeleton)
    model = SpatioTemporalModel(adj, num_joints_in=17, in_features=2, num_joints_out=17,
                                filter_widths=[3, 3, 3], channels=128)
    model = model.cuda()

    model_params = 0

    for parameter in model.parameters():
        model_params += parameter.numel()

    print('INFO: Trainable parameter count:', model_params)
    input = torch.randn(2, 27, 17, 2)
    input = input.cuda()

    # summary(model, (27, 15, 2))
    output = model(input)
    print(output.shape)
