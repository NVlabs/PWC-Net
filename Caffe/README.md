Instructions:

#### Testing (Inference)

1. Please download the code from https://github.com/xmfbit/flownet2 or https://github.com/lmb-freiburg/flownet2 and follow the instructions there to compile the code. (If you encounter problems in compling the code, Please contact deqings@nvidia.com for a docker image)
2. Put warp_layer.cu and warp_layer.cpp (in ./warping_code) to src/caffe/layers and warp_layer.hpp (in ./warping_code) to include/caffe/layers, recompile.
3. Run python ./proc_images.py [img1.txt img2.txt out.txt]. Please compare your results with ./tmp/reference_frame_0010_forward.flo and ./tmp/reference_frame_0011_backward.flo.
4. [TOO ADD] Modify the code and data directory in run_rob_test.py (lines 17-21); make sure that the ROB test images are in your disk.

The program assumes that images to process are of the same size.


#### Benchmarking time

Please modify the caffe directory below

*KITTI resolution*
YOUR_DIRECTORY/flownet2/build/tools/caffe.bin time -model ./benchmark_time/pwc_net_1280_384_batch1.tpl.prototxt -weights ./model/pwc_net.caffemodel -iterations 100 -gpu 0;

*Middlebury "Urban" resolution (accounting for the x2 image resizing)*
YOUR_DIRECTORY/flownet2/build/tools/caffe.bin time -model ./benchmark_time/pwc_net_1280_960_batch1.tpl.prototxt -weights ./model/pwc_net.caffemodel -iterations 100 -gpu 0;

*Sintel resolution*
YOUR_DIRECTORY/flownet2/build/tools/caffe.bin time -model ./benchmark_time/pwc_net_1024_448_batch1.tpl.prototxt -weights ./model/pwc_net.caffemodel -iterations 100 -gpu 0;

*HD1K resolution (requires a GPU with 16G+ memory, such as  NVIDIA Tesla Volta 100)*
YOUR_DIRECTORY/flownet2/build/tools/caffe.bin time -model ./benchmark_time/pwc_net_2560_1088_batch1.tpl.prototxt -weights ./model/pwc_net.caffemodel -iterations 100 -gpu 0;


#### Training 

1.Please download the FlyingChairs dataset from https://lmb.informatik.uni-freiburg.de/resources/datasets, make the LMDB file, modify the local directory in ./model/train.prototxt. 
2.Modify the local directory in  ./train.py and Run ./train.py

### Method description
The model here is PWC-Net with a larger feature pyramid extractor (PWC-Net-feature-uparrow, second row in Table5(a) of Our CVPR 2018 paper below).

#### Paper & Citation
Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume." CVPR 2018. 
Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume." arXiv preprint arXiv:1709.02371(https://arxiv.org/abs/1709.02371), 2017.
Project webpage: http://research.nvidia.com/publication/2018-02_PWC-Net:-CNNs-for
https://github.com/NVlabs/PWC-Net
 

If you use PWC-Net, please cite the following paper: 
```
@InProceedings{Sun2018PWC-Net,
  author    = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
  title     = {{PWC-Net}: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume},
  booktitle = CVPR,
  year      = {2018},
}
```
or the arXiv paper
```
@article{sun2017pwc,
  author={Sun, Deqing and Yang, Xiaodong and Liu, Ming-Yu and Kautz, Jan},
  title={{PWC-Net}: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume},
  journal={arXiv preprint arXiv:1709.02371},
  year={2017}
}
```

#### Contact
Deqing Sun (deqings@nvidia.com)

#### License 
Copyright (C) 2018 NVIDIA Corporation. All rights reserved. 
