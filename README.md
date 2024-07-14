# UCL
This is the PyTorch implementation for "Uncertainty-Guided Contrastive Learning for Weakly Supervised Point Cloud Segmentation".

This code and framework are  implemented on [PointNeXt](https://github.com/guochengqian/PointNeXt).

>Three-dimensional point cloud data are widely used
in many fields, as they can be easily obtained and contain rich
semantic information. Recently, weakly supervised segmentation
has attracted lots of attention, because it only requires very few
labels, thus reducing time-consuming and expensive data anno-
tation efforts for huge amounts of point cloud data. The existing
approaches typically adopt softmax scores from the last layer
as the confidence for selecting high-confident point predictions.
However, such approaches can ignore the potential value of a
large number of low-confidence point predictions under tradi-
tional metrics. In this work, we propose an uncertainty-guided
contrastive learning (UCL) framework for weakly supervised
point cloud segmentation. A novel uncertainty metric based on
prototype entropy (PE) is presented to estimate the reliability
of model predictions. With this metric, we propose a negative
contrastive learning module exploiting negative pseudo-labels
of predictions with low reliability and an active contrastive
learning module enhancing feature learning of segmentation
models by predictions with high reliability. We also propose a
generic multiscale feature perturbation method to expand a wider
perturbation space. Extensive experimental results on indoor
and outdoor point cloud datasets demonstrate that the proposed
method achieves competitive performance.

## Environment and Datasets
This codebase was tested with the following environment configurations.

* Ubuntu 22.04
* Python 3.7
* CUDA 11.3
* Pytorch 1.10.1

Please refer to PointNeXt to install other required packages and download datasets.

## Usage
To stabilize the training process, the first step is to train using only labeled data. Then, set this pre-trained model path in cfg_s3dis.yaml file to conduct weakly supervised training.
````
run ./UCL/main.py
````

## Citation
````
@article{yao2024uncertainty,
  title={Uncertainty-guided Contrastive Learning for Weakly Supervised Point Cloud Segmentation},
  author={Yao, Baochen and Dong, Li and Qiu, Xiaojie and Song, Kangkang and Yan, Diqun and Peng, Chengbin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
````


## Acknowledgement
The code is built on PointNeXt. We thank the authors for sharing the code.

