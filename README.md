[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
## PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume

### License
Copyright (C) 2018 NVIDIA Corporation. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


### Usage

Pleaese follow the README in the Caffe and PyTorch folders to run the code. 

One thing to note is that the PyTorch version does not fully reproduce the Caffe results (~3% performance drop on Sintel). There are differences in implementation between Caffe and PyTorch, such as image resizing and I/O. 


### Paper & Citation
Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume." CVPR 2018. 
Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume." arXiv preprint arXiv:1709.02371(https://arxiv.org/abs/1709.02371), 2017.
Project webpage: http://research.nvidia.com/publication/2018-02_PWC-Net:-CNNs-for
https://github.com/NVlabs/PWC-Net
 

If you use PWC-Net, please cite the following paper: 

@InProceedings{Sun2018PWC-Net,
  author    = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
  title     = {{PWC-Net}: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume},
  booktitle = CVPR,
  year      = {2018},
}

or the arXiv paper

@article{sun2017pwc,
  author={Sun, Deqing and Yang, Xiaodong and Liu, Ming-Yu and Kautz, Jan},
  title={{PWC-Net}: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume},
  journal={arXiv preprint arXiv:1709.02371},
  year={2017}
}


### Contact
Deqing Sun (deqings@nvidia.com)

