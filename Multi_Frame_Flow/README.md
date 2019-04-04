# A Fusion Approach for Multi-Frame Optical Flow Estimation
Zhile Ren, Orazio Gallo, Deqing Sun, Ming-Hsuan Yang, Erik B. Sudderth, and Jan Kautz

<p align="center"> <img src="./.github/teaser.png" width="100%"> </p>

Abstract
-------------------
At the time of this publication, top-performing optical flow estimation methods only take pairs of consecutive frames into account. While elegant and appealing, the idea of using more than two frames has not yet produced state-of-the-art results. We present a simple, yet effective fusion approach for multi-frame optical flow that benefits from longer-term temporal cues. Our method first warps the optical flow from previous frames to the current, thereby yielding multiple plausible estimates. It then fuses the complementary information carried by these estimates into a new optical flow field. At the time of writing, our method ranks first among published results in the MPI Sintel and KITTI 2015 benchmarks.

#### Installation
The code was developed using Python 2.7 & PyTorch 0.2 & CUDA 8.0. There may be a problem related to software versions. To fix the problem, you may look at the implementation in PWCNet.py and replace the syntax to match the new PyTorch environment. 
Install correlation package (if you haven't installed this package before):

- Go to external_packages/correlation-pytorch-master/, follow the instruction in the readme.MD file there
- You might have to run the command in make_cuda.sh line by line

Additionally, we provide a simple installation script using Anaconda for the above steps:
```
# setup environment
conda create -n pwcnet_test python=2.7 anaconda
conda activate pwcnet_test
# install pytorch and other dependencies
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
pip install torchvision visdom dominate opencv-python cffi
# install external packages 
cd external_packages/correlation-pytorch-master/
bash make_cuda.sh
cd ../channelnorm_package/
bash make.sh
cd ../../
```

#### Test
-  Test the code: execute script_pwc.py [image1_filename] [image2_filename] [save_flow_filename], e.g. 
	 ```python script_pwc.py './data/frame_0010.png' './data/frame_0011.png' './tmp/frame_0010.flo'```
-  Please download the fine-tuned weights of PWC-Net on Sintel at https://research.nvidia.com/sites/default/files/pubs/2019-01_A-Fusion-Approach/pwc_net.pth_.tar_.zip

#### Test Multi-Frame Flow using PWC-Net
-  Test the code: execute script_pwc_fusion.py [image1_filename] [image2_filename] [image3_filename] [save_flow2to3_filename], e.g. 
	 ```python script_pwc_fusion.py ./data/frame_0009.png ./data/frame_0010.png ./data/frame_0011.png ./tmp/frame_0010_fusion.flo```
- fusion_net.pth.tar is the fine-tuned weight on MPI Sintel

#### Stuff to Remember:
- Following the Caffe convention, the RGB channel order is reversed to BGR
- Following the implementation of the PWC model we use, after dividing by 255.0, no further normalization is performed
- The estimated flow needs to be scaled by 20.0 because, in training, the GT flow is scaled down by 20.0

#### Note
- The PyTorch code does not fully reproduce the results of the Caffe code because of differences in implementation details, such as resizing and image I/O. See also the PWC-Net main page for more information.
- The average EPE at MPI Sintel is 1.83 (clean) and 2.31 (final). For the official Caffe implementation, the average EPE is 1.81 (clean) and 2.29 (final)

#### Acknowledgments
- Thanks to Dr. Jinwei Gu for helping with writing the code and conversion of the PWC-Net Caffe model into PyTorch
- Thanks to Dr. Fitsum Reda for providing the wrapper to the correlation code

#### Paper & Citation
The paper can be found on Arxiv: <a href>https://arxiv.org/pdf/1810.10066.pdf</a>

If you use this code please cite our paper:
```
@inproceedings{ren2018fusion,
  title={A Fusion Approach for Multi-Frame Optical Flow Estimation},
  author={Ren, Zhile and Gallo, Orazio and Sun, Deqing and Yang, Ming-Hsuan and Sudderth, Erik B and Kautz, Jan},
  booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2019}
}
```

#### Contact
Orazio Gallo (ogallo@nvidia.com); Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)

#### License 
Copyright (C) 2018 NVIDIA Corporation. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (<a href>https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode</a>). 



