#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda-8.0

#cd correlation-pytorch/correlation_package/src
echo "Compiling correlation layer kernels by nvcc..."

# TODO (JEB): Check which arches we need
nvcc -c -o /external_packages/correlation-pytorch-master/correlation-pytorch/correlation_package/src/corr_cuda_kernel.cu.o /external_packages/correlation-pytorch-master/correlation-pytorch/correlation_package/src/corr_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
nvcc -c -o /external_packages/correlation-pytorch-master/correlation-pytorch/correlation_package/src/corr1d_cuda_kernel.cu.o /external_packages/correlation-pytorch-master/correlation-pytorch/correlation_package/src/corr1d_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

#cd ../../
cd /external_packages/correlation-pytorch-master/correlation-pytorch/
pip install .
