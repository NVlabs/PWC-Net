#! /usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import models
import pdb

def writeFlowFile(filename, uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())

# Default values if the function is called with no input parameters
im0_fn = 'data/frame_0009.png';
im1_fn = 'data/frame_0010.png';
im2_fn = 'data/frame_0011.png';
flow_fn = './tmp/frame_0010_fusion.flo';

if len(sys.argv) > 1:
	im0_fn = sys.argv[1]
if len(sys.argv) > 2:
    im1_fn = sys.argv[2]
if len(sys.argv) > 3:
    im2_fn = sys.argv[3]
if len(sys.argv) > 4:
    flow_fn = sys.argv[4]

pwc_model_fn = './pwc_net.pth.tar';

im_all = [imread(img) for img in [im0_fn, im1_fn, im2_fn]]
im_all = [im[:, :, :3] for im in im_all]

# rescale the image size to be multiples of 64
divisor = 64.
H = im_all[0].shape[0]
W = im_all[0].shape[1]

H_ = int(ceil(H/divisor) * divisor)
W_ = int(ceil(W/divisor) * divisor)
for i in range(len(im_all)):
	im_all[i] = cv2.resize(im_all[i], (W_, H_))

for _i, _inputs in enumerate(im_all):
	im_all[_i] = im_all[_i][:, :, ::-1]
	im_all[_i] = 1.0 * im_all[_i]/255.0
	
	im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
	im_all[_i] = torch.from_numpy(im_all[_i])
	im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
	im_all[_i] = im_all[_i].float()
    
# compute two frame flows
input_01 = [im_all[0].cuda(), im_all[1].cuda()]
input_01_var = torch.autograd.Variable(torch.cat(input_01,1), volatile=True)

input_12 = [im_all[1].cuda(), im_all[2].cuda()]
input_12_var = torch.autograd.Variable(torch.cat(input_12,1), volatile=True)

input_10 = [im_all[1].cuda(), im_all[0].cuda()]
input_10_var = torch.autograd.Variable(torch.cat(input_10,1), volatile=True)


net = models.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()
for p in net.parameters():
    p.requires_grad = False

cur_flow = net(input_12_var) * 20.0
prev_flow = net(input_01_var) * 20.0
prev_flow_back = net(input_10_var) * 20.0

# perfom flow fusion
net_fusion = models.netfusion_custom(path="./fusion_net.pth.tar",
                                     div_flow=20.0, 
                                     batchNorm=False)
net_fusion = net_fusion.cuda()
net_fusion.eval()
for p in net_fusion.parameters():
    p.requires_grad = False

upsample_layer = torch.nn.Upsample(scale_factor=4, mode='bilinear')

cur_flow = upsample_layer(cur_flow)
prev_flow = upsample_layer(prev_flow)
prev_flow_back = upsample_layer(prev_flow_back)
input_var_cat = torch.cat((input_12_var, cur_flow, prev_flow, prev_flow_back), dim=1)
flo = net_fusion(input_var_cat)

flo = flo[0] * 20.0
flo = flo.cpu().data.numpy()

# scale the flow back to the input size 
flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
u_ = cv2.resize(flo[:,:,0],(W,H))
v_ = cv2.resize(flo[:,:,1],(W,H))
u_ *= W/ float(W_)
v_ *= H/ float(H_)
flo = np.dstack((u_,v_))

writeFlowFile(flow_fn, flo)
