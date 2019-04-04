This repository contains a custom pytorch package that adds a module and functional interface for the correlation layer described in "FlowNet: Learning Optical Flow with Convolutional Networks" (https://arxiv.org/abs/1504.06852)

To install:

1. Run `pip install cffi`

2. Run `make_cuda.sh`

3. Run `python setup.py build install`

4. (optional) Run `python test/test.py`


#### Acknowledgement
- Thanks to Dr. Fitsum Reda for providing the wrapper to the correlation code

If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper:
```
@misc{flownet2-pytorch,
  author = {Fitsum Reda and Robert Pottorff and Jon Barker and Bryan Catanzaro},
  title = {flownet2-pytorch: Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVIDIA/flownet2-pytorch}}
}
```
