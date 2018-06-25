[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
## PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume

![](flow.gif)

### License
Copyright (C) 2018 NVIDIA Corporation. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


### Usage

For Caffe users, please refer to [Caffe/README.md](Caffe/README.md).

For PyTorch users, please refer to [PyTorch/README.md](PyTorch/README.md)

Note that, currently, the PyTorch implementation is inferior to the Caffe implementation (~3% performance drop on Sintel). These are due to differences in implementation between Caffe and PyTorch, such as image resizing and I/O. 

### Network Architecture

PWC-Net fuses several classic optical flow estimation techniques, including image pyramid, warping, and cost volume, in an end-to-end trainable deep neural networks for achieving state-of-the-art results.

![](network.png)


### Paper & Citation
[Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume." CVPR 2018 or arXiv:1709.02371](https://arxiv.org/abs/1709.02371)


[Project page link](http://research.nvidia.com/publication/2018-02_PWC-Net:-CNNs-for)

 

If you use PWC-Net, please cite the following paper: 

@InProceedings{Sun2018PWC-Net,
  author    = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
  title     = {{PWC-Net}: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume},
  booktitle = CVPR,
  year      = {2018},
}


### Contact
Deqing Sun (deqings@nvidia.com)

