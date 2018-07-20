
#### Installation
The code was developed using Python 2.7 & PyTorch 0.2 & CUDA 8.0. There may be a problem related to software versions. To fix the problem, you may look at the implementation in PWCNet.py and replace the syntax to match the new pytorch environment. 
Install correlation package (if you haven't installed this package before):
- Go to external_packages/correlation-pytorch-master/, follow the instruction in the readme.MD file there
- You might have to run the command in make_cuda.sh line by line

#### Test
-  Test the code: execute script_pwc.py [image1_filename] [image2_filename] [save_flow_filename], e.g. 
	 ```python script_pwc.py './data/frame_0010.png' './data/frame_0011.png' './tmp/frame_0010.flo'```
-  pwc_net_chairs.pth.tar is the pretrained weight using flyingthings3D dataset
-  pwc_net.pth.tar is the fine-tuned weight on MPI Sintel


#### Stuffs that may confuse you:
- the RGB channel is reversed to BGR, because this is what Caffe does
- after dividing by 255.0, no further normalization is conducted, because this particular PWC model in Caffe don't perform any image normalizations
- estimated flow should multiply by 20.0, because in training, the GT flow is divided by 20.0


#### Note
- The PyTorch code does not fully reproduce the results by the Caffe code because of differences in implementation details, such as resizing and image I/O. Please use the Caffe code to reproduce the PWC-Net results in the CVPR paper and the ROB challenge
- The average EPE at MPI Sintel is 1.83 (clean) and 2.31 (final). For the official Caffe implementation, the average EPE is 1.81 (clean) and 2.29 (final)


#### Acknowledgement
- Thanks to Dr. Jinwei Gu and Dr. Zhile Ren for writing the PyTorch code and converting the Caffe model into PyTorch
- Thanks to Dr. Fitsum Reda for providing the wrapper to the correlation code

#### Paper & Citation
Deqing Sun, Xiaodong Yang, Ming-Yu Liu, Jan Kautz. PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume. CVPR 2018 Oral. 
Sun, Deqing, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume." arXiv preprint arXiv:1709.02371(https://arxiv.org/abs/1709.02371), 2017.
Project webpage: http://research.nvidia.com/publication/2018-02_PWC-Net:-CNNs-for
 

If you use PWC-Net, please cite the following paper: 
```
@InProceedings{Sun2018PWC-Net,
  author    = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
  title     = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
  booktitle = CVPR,
  year      = {2018},
}
```
or the arXiv paper
```
@article{sun2017pwc,
  author={Sun, Deqing and Yang, Xiaodong and Liu, Ming-Yu and Kautz, Jan},
  title={{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
  journal={arXiv preprint arXiv:1709.02371},
  year={2017}
}
```

#### Contact
Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)

#### License 
Copyright (C) 2018 NVIDIA Corporation. All rights reserved. 



