# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def get_path(cur_file):
    cur_dir = osp.dirname(osp.realpath(cur_file))
    pre_dir = osp.join(cur_dir, '..')
    project_root = osp.abspath(osp.join(cur_dir, '../../../../'))
    chk_root = osp.join(project_root, 'checkpoint/')
    data_root = osp.join(project_root, 'data/')
    lib_root = osp.join(project_root, 'lib/')
    output_root = osp.join(project_root, 'output/')

    return pre_dir, cur_dir, chk_root, data_root, lib_root, output_root


this_dir = osp.dirname(osp.realpath(__file__))

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
