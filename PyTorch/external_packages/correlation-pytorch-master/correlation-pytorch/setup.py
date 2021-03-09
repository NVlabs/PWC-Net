#!/usr/bin/env python

import os

from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

this_file = os.path.dirname(__file__)


corr_source = ['correlation_package/src/corr.c', 'correlation_package/src/corr1d.c']
corr_includes = ['correlation_package/src/']

if torch.cuda.is_available():
    print('Including CUDA code.')
    ext_fnct = CUDAExtension
    corr_source += ['correlation_package/src/corr_cuda.c', 'correlation_package/src/corr1d_cuda.c']
    corr_source += ['correlation_package/src/corr_cuda_kernel.cu', 'correlation_package/src/corr1d_cuda_kernel.cu']
else:
    ext_fnct = CppExtension

setup(
    name="correlation_package",
    version="0.1",
    description="Correlation layer from FlowNetC",
    url="https://github.com/jbarker-nvidia/pytorch-correlation",
    author="Jon Barker",
    author_email="jbarker@nvidia.com",
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py
    ext_package="",
    # Extensions to compile
    ext_modules=[
        ext_fnct(
            'correlation_package._ext.corr',
            corr_source, include_dirs=corr_includes,
            extra_compile_args={'cxx': ['-std=c++14']},),
    ],
    cmdclass={'build_ext': BuildExtension}
)
