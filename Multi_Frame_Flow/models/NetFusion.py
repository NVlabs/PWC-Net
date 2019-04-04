#! /usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory 
from correlation_package.modules.corr import Correlation 
import numpy as np
import torch.nn.init as nn_init

__all__ = [
 'NetFusion'
]


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                      stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                      stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                      stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                      stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )
    

class NetFusion(nn.Module):
    def __init__(self, batchNorm=False, inPlanes=11):
        super(NetFusion,self).__init__()

        self.batchNorm = batchNorm
        self.conv0   = conv(self.batchNorm,  inPlanes,   64)
        self.conv1   = conv(self.batchNorm,  64,   64, stride=2)
        self.conv1_1 = conv(self.batchNorm,  64,   128)
        self.conv2   = conv(self.batchNorm,  128,  128, stride=2)
        self.conv2_1 = conv(self.batchNorm,  128,  128)

        self.deconv1 = deconv(128,32)
        self.deconv0 = deconv(162,16)

        self.inter_conv1 = i_conv(self.batchNorm,  162,   32)
        self.inter_conv0 = i_conv(self.batchNorm,  82,   16)

        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)

        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        flow2       = self.predict_flow2(out_conv2)
        flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        
        concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1       = self.predict_flow1(out_interconv1)
        flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        
        concat0 = torch.cat((out_conv0,out_deconv0,flow1_up),1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0       = self.predict_flow0(out_interconv0)

        return flow0
