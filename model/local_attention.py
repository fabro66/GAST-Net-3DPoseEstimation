from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SemCHGraphConv(nn.Module):
    """
    Semantic channel-wise graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=False):
        super(SemCHGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj.unsqueeze(0).repeat(out_features, 1, 1)
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(out_features, len(self.m[0].nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # input: (B, T, J, C)
        h0 = torch.matmul(input, self.W[0]).unsqueeze(2).transpose(2, 4)  # B * T * C * J * 1
        h1 = torch.matmul(input, self.W[1]).unsqueeze(2).transpose(2, 4)  # B * T * C * J * 1

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)  # C * J * J
        adj[self.m] = self.e.view(-1)
        adj = F.softmax(adj, dim=2)

        E = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        E = E.unsqueeze(0).repeat(self.out_features, 1, 1)  # C * J * J

        output = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
        output = output.transpose(2, 4).squeeze(2)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LocalGraph(nn.Module):
    def __init__(self, adj, input_dim, output_dim, dropout=None):
        super(LocalGraph, self).__init__()
        
        num_joints = adj.shape[0]

        # Human3.6M
        if num_joints == 17:
            distal_joints = [3, 6, 10, 13, 16]
            joints_left = [4, 5, 6, 11, 12, 13]
            joints_right = [1, 2, 3, 14, 15, 16]

        # Human3.6M detected from Stacked Hourglass
        elif num_joints == 16:
            distal_joints = [3, 6, 9, 12, 15]
            joints_left = [4, 5, 6, 10, 11, 12]
            joints_right = [1, 2, 3, 13, 14, 15]

        # HumanEva
        elif num_joints == 15:
            distal_joints = [4, 7, 10, 13]
            joints_left = [2, 3, 4, 8, 9, 10]
            joints_right = [5, 6, 7, 11, 12, 13]

        # Human3.6M including toe keypoints
        elif num_joints == 19:
            distal_joints = [3, 4, 7, 8, 12, 15, 18]
            joints_left = [5, 6, 7, 8, 13, 14, 15]
            joints_right = [1, 2, 3, 4, 16, 17, 18]

        else:
            raise KeyError("The dimension of adj matrix is wrong!")

        adj_sym = torch.zeros_like(adj)
        for i in range(num_joints):
            for j in range(num_joints):
                if i == j:
                    adj_sym[i][j] = 1
                if i in joints_left:
                    index = joints_left.index(i)
                    adj_sym[i][joints_right[index]] = 1.0
                if i in joints_right:
                    index = joints_right.index(i)
                    adj_sym[i][joints_left[index]] = 1.0

        adj_1st_order = adj.matrix_power(1)
        for i in np.arange(num_joints):
            if i in distal_joints:
                adj_1st_order[i] = 0

        adj_2nd_order = adj.matrix_power(2)
        for i in np.arange(num_joints):
            if i not in distal_joints:
                adj_2nd_order[i] = 0

        adj_con = adj_1st_order + adj_2nd_order

        self.gcn_sym = SemCHGraphConv(input_dim, output_dim, adj_sym)
        self.bn_1 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.gcn_con = SemCHGraphConv(input_dim, output_dim, adj_con)
        self.bn_2 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.relu = nn.ReLU()

        self.cat_conv = nn.Conv2d(2*output_dim, output_dim, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(output_dim, momentum=0.1)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input):
        # x: (B, T, K, C)
        x = self.gcn_sym(input)
        y = self.gcn_con(input)

        # x: (B, T, K, C) --> (B, C, T, K)
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        x = self.relu(self.bn_1(x))
        y = self.relu(self.bn_2(y))

        output = torch.cat((x, y), dim=1)
        output = self.cat_bn(self.cat_conv(output))

        if self.dropout is not None:
            output = self.dropout(self.relu(output))
        else:
            output = self.relu(output)
        output = output.permute(0, 2, 3, 1)

        return output
