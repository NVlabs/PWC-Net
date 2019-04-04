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
import sys
sys.path.append("../")
sys.path.append("./external_packages/")
from channelnorm_package.modules.channelnorm import ChannelNorm
import pdb
import models
import NetFusion
import torch.nn.init as nn_init


__all__ = [
 'netfusion_custom'
]

    
class NetFusion_custom(nn.Module):
    def __init__(self, div_flow = 20.0, batchNorm=False):        
        super(NetFusion_custom,self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow

        self.fusion = NetFusion.NetFusion(batchNorm=self.batchNorm, inPlanes=9)
        self.channelnorm = ChannelNorm()

    def warp_back(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()

        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.999] = 0
        mask[mask>0] = 1
        
        return output*mask
            
    def forward(self,x):        
        im1 = x[:, :3,  :, :]
        im2 = x[:, 3:6, :, :]
        cur_flow = x[:, 6:8, :, :]
        prev_flow = x[:, 8:10, :, :]
        prev_flow_back = x[:, 10:12, :, :]

        prev_flow = self.warp_back(prev_flow, prev_flow_back) #warp 0->1 to frame 1
        
        im2_warp_backward_cur_flow = self.warp_back(im2, cur_flow)  # im2 1->2 im1
        im2_warp_backward_prev_flow = self.warp_back(im2, prev_flow) # im2 wapred 1->2 im1


        mask_warp_cur_flow = torch.autograd.Variable(torch.ones(x.size()[0], 1, x.size()[2], x.size()[3]).float()).cuda()
        mask_warp_prev_flow = torch.autograd.Variable(torch.ones(x.size()[0], 1, x.size()[2], x.size()[3]).float()).cuda()

        mask_warp_cur_flow = self.warp_back(mask_warp_cur_flow, cur_flow)
        mask_warp_prev_flow = self.warp_back(mask_warp_prev_flow, prev_flow)

        
        cur_flow = cur_flow / self.div_flow
        prev_flow = prev_flow / self.div_flow
        
        norm_cur_flow = self.channelnorm(cur_flow)
        norm_prev_flow = self.channelnorm(prev_flow)

        diff_im1_cur_flow = self.channelnorm(im1-im2_warp_backward_cur_flow)
        diff_im1_prev_flow = self.channelnorm(im1-im2_warp_backward_prev_flow)

        diff_im1_cur_flow_comp = 0.5 - diff_im1_cur_flow
        diff_im1_cur_flow_comp[mask_warp_cur_flow>0] = 0
        diff_im1_prev_flow_comp = 0.5 - diff_im1_prev_flow
        diff_im1_prev_flow_comp[mask_warp_prev_flow>0] = 0


        diff_im1_cur_flow = diff_im1_cur_flow + diff_im1_cur_flow_comp
        diff_im1_prev_flow = diff_im1_prev_flow + diff_im1_prev_flow_comp


        concat_feat = torch.cat((im1, cur_flow, prev_flow, diff_im1_cur_flow, diff_im1_prev_flow), dim=1)
        
        flow_new = self.fusion(concat_feat)
        
        return flow_new


def netfusion_custom(path=None, div_flow = 20.0, batchNorm=False):
    
    model = NetFusion_custom(div_flow = div_flow, batchNorm=batchNorm)

    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)

    return model
