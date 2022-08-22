#!/usr/bin/env bash

cd correlation-pytorch/correlation_package/src
echo "Compiling correlation layer kernels by nvcc..."

cd ../../
python setup.py build install --user
