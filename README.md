# PCFEA

This is the official PyTorch implementation of our paper:

> **[Progressive Classifier and Feature Extractor Adaptation for Unsupervised Domain Adaptation on Point Clouds](https://arxiv.org/abs/2303.01276)**
> *In European Conference on Computer Vision (ECCV), 2024*


> **Abstract.** 
> Unsupervised domain adaptation (UDA) is a critical challenge in the field of point cloud analysis. Previous works tackle the problem either by feature extractor adaptation to enable a shared classifier to distinguish domain-invariant features, or by classifier adaptation to evolve the classifier to recognize target-styled source features to increase its adaptation ability. However, by learning domain-invariant features, feature extractor adaptation methods fail to encode semantically meaningful target-specific information, while classifier adaptation methods rely heavily on the accurate estimation of the target distribution. In this work, we propose a novel framework that deeply couples the classifier and feature extractor adaption for 3D UDA, dubbed Progressive Classifier and Feature Extractor Adaptation (PCFEA). Our PCFEA conducts 3D UDA from two distinct perspectives: macro and micro levels. On the macro level, we propose a progressive target-styled feature augmentation (PTFA) that establishes a series of intermediate domains to enable the model to progressively adapt to the target domain. Throughout this process, the source classifier is evolved to recognize target-styled source features (\ie, classifier adaptation). On the micro level, we develop an intermediate domain feature extractor adaptation (IDFA) that performs a compact feature alignment to encourage the target-styled feature extraction gradually. In this way, PTFA and IDFA can mutually benefit each other: IDFA contributes to the distribution estimation of PTFA while PTFA constructs smoother intermediate domains to encourage an accurate feature alignment of IDFA. We validate our method on popular benchmark datasets, where our method achieves new state-of-the-art performance.



## Getting Started

### Installation

Please follow the steps in the requirements.txt file to prepare the environment.


### Dataset:

Our code supports PointDA-10 dataset and GraspNetPC-10 dataset.

- Please download PointDA-10 dataset at https://drive.google.com/file/d/1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J/view?usp=sharing.

- Please download GraspNetPC-10 dataset at https://drive.google.com/file/d/1VVHmsSToFMVccge-LsYJW67IS94rNxWR/view?usp=sharing.


Please unzip the datasets and modify the dataset path in configuration files.

For example, if you put your data under the data folder like this, you can directly bash our run_test.sh file to run the code.
```

├── data
    ├── GraspNetPointClouds
        ├── test
        └── train
    └── PointDA_data
        ├── modelnet
        ├── scannet
        └── shapenet

├── PCFEA
    ├── data
        ├── dataloader_XXXX.py
        ├── ....
        └── grasp_datautils
  	├── log
  		├── XXX.txt
  		└── XXX.txt
    ├── models 
        ├── model.py
        └── pointnet_util.py
    ├── utils
        ├── log_SPST.py
        ├── ....
        └── trans_norm.py
    ├── augmentation.py
    ├── ....
    └── train_PCFEA_cls.py

```



## Usage

We tried many different methods before finally coming up with our PCFEA. So, we also provide the codes of some of the methods we tried, so the config may seem a bit cumbersome. 
If you want to reproduce our results, please directly bash run_test.sh.

To run with different settings, please modify the settings in the sh file.

We have uploaded the log files in the log folder.

Note that all of our experiments are tested on 2080Ti, A5000 or 3090.



## Citation

If you find these projects useful, please consider citing our paper.




## Acknowledgement

We thank [GAST](https://github.com/zou-longkun/GAST), [MLSP](https://github.com/VITA-Group/MLSP), [DefRec_and_PCM](https://github.com/IdanAchituve/DefRec_and_PCM), [PointDAN](https://github.com/canqin001/PointDAN), [ImplicitPCDA](https://github.com/Jhonve/ImplicitPCDA), [DGCNN](https://github.com/WangYueFt/dgcnn), [PointNet](https://github.com/charlesq34/pointnet), [PointNet++](https://github.com/charlesq34/pointnet2) and other relevant works for their amazing open-sourced projects!