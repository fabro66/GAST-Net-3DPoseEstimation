# A Graph Attention Spatio-temporal Convolutional Networks for 3D Human Pose Estimation in Video (GAST-Net)

### News
* [2020/08/14] We achieve real-time 3D pose estimation. [[video]](https://www.bilibili.com/video/BV18f4y197R7/)
* [2020/10/15] We achieve real-time online 3D skeleton-based action recognition with a single RGB camera. [[video]](https://www.bilibili.com/video/BV1e54y1i7iT/?spm_id_from=333.788.videocard.0)
* [2020/11/16] We release codes on how to generate 3D poses and animation from a custom video. [[INFERENCE.md]](./INFERENCE_EN.md)
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
<img align=center>![GAST-Net Framework](./image/framework.png)


### Real-time estimation
<img align=center>![Realtime Estimation](./image/RealtimeEstimation.gif)

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python >=3.6
- PyTorch >= 1.0.1
- matplotlib
- numpy

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
python trainval.py -e 60 -k cpn_ft_h36m_dbb -arc 3,3,3 -drop 0.05 -b 128
```

For HumanEva:
```
python trainval.py -d humaneva15 -e 200 -k detectron_pt_coco -d humaneva15 -arc 3,3,3 -drop 0.5 -b 32 -lr 0.98 -str Train/S1,Train/S2,Train/S3 -ste Validate/S1,Validate/S2,Validate/S3 -a Walk,Jog,Box --by-subject
```

To test on Human3.6M, run:
```
python trainval.py -k cpn_ft_h36m_dbb -arc 3,3,3 -c checkpoint --evaluate epoch_60.bin
```

To test on HumanEva, run:
```
python trainval.py -k detectron_pt_coco -arc 3,3,3 -str Train/S1,Train/S2,Train/S3 -ste Validate/S1,Validate/S2,Validate/S3 -a Walk,Jog,Box --by-subject -c checkpoint --evaluate epoch_60.bin
```

### Download our pretrained models
```
cd root_path
mkdir checkpoint output
cd checkpoint
mkdir gastnet
```
    -checkpoint\gastnet\
                27_frame_model.bin
                81_frame_model.bin

* Google Drive:
> [27 receptive field model](https://drive.google.com/file/d/1vh29QoxIfNT4Roqw1SuHDxxKex53xlOB/view?usp=sharing)

> [81 receptive field model](https://drive.google.com/file/d/12n-CyDhImxwHmakfA24n5Nz7J6QXj83f/view?usp=sharing)
  
* Baidu Yun(Extract code: kuxf):
> [27 receptive field model](https://pan.baidu.com/s/1tLCCm5l7izffziaNERGp0w)

> [81 receptive field model](https://pan.baidu.com/s/1tLCCm5l7izffziaNERGp0w)

### Reconstruct 3D poses from 2D keypoints
Reconstruct 3D poses from 2D keypoints estimated from 2D detector (Mask RCNN, HRNet and OpenPose et al), and visualize it.

If you want to reproduce the baseball example, please run the following code:
```
python reconstruction.py
```

or run more detailed parameter settings:
```
python reconstruction.py -f 27 -w 27_frame_model.bin -k ./data/keypoints/baseball.json -vi ./data/video/baseball.mp4 -vo ./output/baseball_reconstruction.mp4 -kf coco
```
* Reconstructed from YouTube video
![](./image/Baseball.gif)

* Reconstructed from [NTU-RGBD](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset 
![](./image/WalkApart.gif)

### Inference a custom video
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
  author={Liu, Junfa and Rojas, Juan and Liang, Zhijun and Li, Yihui and Guan},
  journal={arXiv preprint arXiv:2003.14179},
  year={2020}
}
```

### Contact
* If you have any questions, please fell free to contact us. (junfaliu2019@gmail.com)