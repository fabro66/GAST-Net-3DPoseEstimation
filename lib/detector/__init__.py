import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(osp.realpath(__file__)), 'yolov3'))
from human_detector import yolo_human_det, load_model
sys.path.pop(0)