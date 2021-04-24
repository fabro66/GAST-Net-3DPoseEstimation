# A Graph Attention Spatio-temporal Convolutional Networks for 3D Human Pose Estimation in Video (GAST-Net)

### News
* [2021/01/28] We update GAST-Net to able to generate 19-joint human poses including body and foot joints. [**[DEMO]**](./image/Baseball_body_foot.gif)
* [2020/11/17] We provide a tutorial on how to generate 3D poses/animation from a custom video. [**[INFERENCE_EN.md]**](./INFERENCE_EN.md)
* [2020/10/15] We achieve online 3D skeleton-based action recognition with a single RGB camera. [**[video]**](https://www.bilibili.com/video/BV1e54y1i7iT/?spm_id_from=333.788.videocard.0)[**[code]**](https://github.com/fabro66/Online-Skeleton-based-Action-Recognition)
* [2020/08/14] We achieve real-time 3D pose estimation. [**[video]**](https://www.bilibili.com/video/BV18f4y197R7/)
### Introduction
Spatio-temporal information is key to resolve occlusion and depth ambiguity in 3D pose estimation. Previous methods have focused on either temporal contexts or local-to-global architectures that embed fixed-length spatio-temporal information. 
To date, there have not been effective proposals to simultaneously and flexibly capture varying spatio-temporal sequences and effectively achieves real-time 3D pose estimation.
In this work, we improve the learning of kinematic constraints in the human skeleton: posture, local kinematic connections, and symmetry by modeling local and global spatial information via attention mechanisms. 
To adapt to single- and multi-frame estimation, the dilated temporal model is employed to process varying skeleton sequences. 
Also, importantly, we carefully design the interleaving of spatial semantics with temporal dependencies to achieve a synergistic effect. 
To this end, we propose a simple yet effective graph attention spatio-temporal convolutional network (GAST-Net) that comprises of interleaved temporal convolutional and graph attention blocks.
Combined with the proposed method, we introduce a real-time strategy for online 3D skeleton-based action recognition with a simple RGB camera.
Experiments on two challenging benchmark datasets (Human3.6M and HumanEva-I) and YouTube videos demonstrate that our approach effectively mitigates depth ambiguity and self-occlusion, generalizes to half upper body estimation, and achieves competitive performance on 2D-to-3D video pose estimation.

* [A Graph Attention Spatio-temporal Convolutional Networks for 3D Human Pose Estimation in Video](https://arxiv.org/abs/2003.14179).
* Project Website: [http://www.juanrojas.net/gast/](http://www.juanrojas.net/gast/) 

### FrameWork
<img align=center>
<img src="./image/framework.png"/>
</div>

* Reconstructed from [NTU-RGBD](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset [**[Introduction]**](./INFERENCE_EN.md) 

<div align=center>
<img src="https://github.com/fabro66/GAST-Net-3DPoseEstimation/blob/master/image/WalkApart.gif" alt=" Two-person 3D human pose estimation"/>
</div>

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python >=3.6
- PyTorch >= 1.0.1
- matplotlib
- numpy
- ffmpeg

### Data preparation
- Download the raw data from [Human3.6M](http://vision.imar.ro/human3.6m) and [HumanEva-I](http://humaneva.is.tue.mpg.de/)
- Preprocess the dadaset in the same way as like [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)
- Then put the preprocessed dataset under the data directory

       -data\
            data_2d_h36m_gt.npz
            data_3d_36m.npz
            data_2d_h36m_cpn_ft_h36m_dbb.npz
            data_2d_h36m_sh_ft_h36m.npz
        
            data_2d_humaneva15_gt.npz
            data_3d_humaneva15.npz
            data_2d_humaneva15_detectron_pt_coco.npz

### Training & Testing
If you want to reproduce the results of our paper, run the following commands.

For Human3.6M:
```
python trainval.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3 -drop 0.05 -b 128
```

For HumanEva:
```
python trainval.py -d humaneva15 -e 200 -k detectron_pt_coco -d humaneva15 -arc 3,3,3 -drop 0.5 -b 32 -lrd 0.98 -str Train/S1,Train/S2,Train/S3 -ste Validate/S1,Validate/S2,Validate/S3 -a Walk,Jog,Box --by-subject
```

To test on Human3.6M, run:
```
python trainval.py -k cpn_ft_h36m_dbb -arc 3,3,3 -c checkpoint --evaluate epoch_60.bin
```

To test on HumanEva, run:
```
python trainval.py -k detectron_pt_coco -arc 3,3,3 -str Train/S1,Train/S2,Train/S3 -ste Validate/S1,Validate/S2,Validate/S3 -a Walk,Jog,Box --by-subject -c checkpoint --evaluate epoch_200.bin
```

### Download our pretrained models from model zoo([GoogleDrive](https://drive.google.com/drive/folders/194Btr2L2FJ7jWaH4c1mpNysvKZOcEb1K?usp=sharing) or [BaiduDrive (ietc)](https://pan.baidu.com/s/1AVPEtpuwLqYjDC3f9Ita0A))
```
cd root_path
mkdir checkpoint output
cd checkpoint
mkdir gastnet
```
    -checkpoint\gastnet\
                27_frame_model.bin
                27_frame_model_toe.bin

### Reconstruct 3D poses from 2D keypoints
Reconstruct 3D poses from 2D keypoints estimated from 2D detector (Mask RCNN, HRNet and OpenPose et al), and visualize it.

If you want to reproduce the baseball example (17 joints, only include body joints), please run the following code:
```
python reconstruction.py
```

If you want to reproduce the baseball example (19 joints, include body and toe joints), please run the following code:
```
python reconstruction.py -w 27_frame_model_toe.bin -n 19 -k ./data/keypoints/baseball_wholebody.json -kf wholebody
```
* Reconstructed from YouTube video
<div align=center>
<img src="https://github.com/fabro66/GAST-Net-3DPoseEstimation/blob/master/image/Baseball.gif" width="640" alt="17-joint 3D human pose estimation"/>
</div>
<div align=center>
<img src="https://github.com/fabro66/GAST-Net-3DPoseEstimation/blob/master/image/Baseball_body_foot.gif" width="640" alt="19-joint 3D human pose estimation">
</div>

### How to generate 3D human poses from a custom video
We provide a tutorial on how to run our model on custom videos. See [INFERENCE.md](./INFERENCE_EN.md) for more details.

### Acknowledgements
This repo is based on 
- [YOLOv3](https://github.com/ayooshkathuria/pytorch-yolo-v3)
- [SORT](https://github.com/abewley/sort)
- [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) 

Thanks to the original authors for their work!

### Reference
If you find our paper and repo useful, please cite our paper. Thanks!

```
@article{liu2020a,
  title={A Graph Attention Spatio-temporal Convolutional Networks for 3D Human Pose Estimation in Video},
  author={Liu, Junfa and Rojas, Juan and Liang, Zhijun and Li, Yihui and Guan, Yisheng},
  journal={arXiv preprint arXiv:2003.14179},
  year={2020}
}
```

### Contact
* If you have any questions, please fell free to contact us. (junfaliu2019@gmail.com)
