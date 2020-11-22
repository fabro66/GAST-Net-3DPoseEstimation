# 基于RGB视频的人体3D姿态估计
语言: [[English]](INFERENCE_EN.md) [[中文]](INFERENCE_CH.md)

本教程，我们将展示如何在RGB视频上进行人体3D姿态估计并可视化。3D姿态重构建采用我们提出的GAST-Net模型，具体细节可见[文章](https://arxiv.org/abs/2003.14179)内容。此教程提供的代码仅适用于实验研究，存在一定的局限性，暂不适合实际的落地应用。

- 功能：可实现RGB视频中的单人和双人3D姿态估计，输出基于人体盆骨基点的3D关节坐标，或者生成动图。

- 工作原理：首先采用YOLOv3和SORT对视频中的行人进行检测和跟踪，然后利用HRNet对检测的行人进行2D姿态估计，最后通过GAST-Net回归生成3D姿态。

<div align=center>
    <img src="./image/input.png" width="200" alt="Input">      <img src="./image/detection_tracking.png" width="200" alt="detection and tracking">      <img src="./image/pose_estimation.png" width="200" alt="2D pose estimation">      <img src="./image/reconstruction.png" width="200" alt="3D reconstruction">
</div>


## 模型下载

- 创建YOLOv3预训练模型仓库，下载预训练模型：
``` cd root_path
    cd checkpoint
    mkdir yolov3
    wget https://pjreddie.com/media/files/yolov3.weights
```


- 创建HRNet预训练模型仓库，下载预训练模型：
``` cd checkpoint
    mkdir hrnet
    cd hrnet
    mkdir pose_coco
```
下载HRNet预训练模型[[pose_hrnet_w48_384x288.pth]](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)，并放进pose_coco文件夹中


- 创建GAST-Net预训练模型仓库，下载预训练模型：
``` cd checkpoint
    mkdir gastnet
```
下载GAST-Net预训练模型[[27_frame_model.bin]](https://pan.baidu.com/s/1tLCCm5l7izffziaNERGp0w),提取密码:kuxf，并放进gastnet文件夹中

```
    ${root_path}
    -- checkpoint
        |-- yolov3
            |-- yolov3.weights
        |-- hrnet
            |-- pose_coco
                |-- pose_hrnet_w48_384x288.pth
        |-- gastnet
            |-- 27_frame_model.bin
```
## 3D姿态动图生成
- 单人3D姿态估计：
```
    python gen_skes.py -v baseball.mp4 -np 1 --animation
```
- 双人3D姿态估计：
```
    python gen_skes.py -v apart.avi -np 2 --animation
```
![baseball](./image/WalkApart.gif)

-- 生成的动图默认放在**output**文件夹中


## 3D姿态骨架文件生成
- 单人3D姿态估计：
```
    python gen_skes.py -v baseball.mp4 -np 1
```
