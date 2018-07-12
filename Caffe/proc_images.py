#!/usr/bin/env python
import time
import glob,os,sys
import subprocess
from math import ceil
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from scipy.misc import imread,imsave

# to output the visualization, please download flow_io.py and viz_flow.py from https://github.com/jswulff/pcaflow/tree/master/pcaflow/utils
# from flow_io import flow_read
# from viz_flow import viz_flow 


# Please modify to your local caffe directory
caffe_bin = '/mnt/zfs/projects/ml_flow/v2x/flownet2/build/tools/caffe.bin'

gpu_id = 0

# =========================================================
def evaluate_model(template, model_filename, img1_filename, img2_filename, save_filenames, vis_save_filenames,  scale_ratio=1.0):
    if not os.path.isfile(caffe_bin):
        print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
        sys.exit(1)

    with open(img1_filename, 'r') as f:
        images1 = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    with open(img2_filename, 'r') as f:
        images2 = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    if len(images1) != len(images2):
        print("Unequal amount of images in the given lists (%d vs. %d)" % (len(images1), len(images2)))
        sys.exit(1)

    list_length = len(images1)
    im1 = imread(images1[0])
    im2 = imread(images2[0])
    if im1.shape != im2.shape:
        print("The image pairs do not have equal sizes!")
        sys.exit(1)       
    width  = im1.shape[1]
    height = im1.shape[0]    
    # print list_length


    # Prepare prototxt
    subprocess.call(['cp', img1_filename, 'tmp/img1.txt'])
    subprocess.call(['cp', img2_filename, 'tmp/img2.txt'])

    # set up width and height for processing
    divisor = 64.
    adapted_width   = ceil(width/divisor*scale_ratio) * divisor
    adapted_height  = ceil(height/divisor*scale_ratio) * divisor        
    rescale_coeff_x = width / adapted_width
    rescale_coeff_y = height / adapted_height
    replacement_list = {
        '$ADAPTED_WIDTH': ('%d' % adapted_width),
        '$ADAPTED_HEIGHT': ('%d' % adapted_height),
        '$TARGET_WIDTH': ('%d' % width),
        '$TARGET_HEIGHT': ('%d' % height),
        '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
        '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y)
    }
    proto = ''
    with open(template, "r") as tfile:
        proto = tfile.read()
    for r in replacement_list:
        proto = proto.replace(r, replacement_list[r])
    with open('tmp/deploy.prototxt', "w") as tfile:
        tfile.write(proto)


    args = [caffe_bin, 'test', '-model', 'tmp/deploy.prototxt',
            '-weights', model_filename,  '-iterations', str(list_length), '-gpu', str(gpu_id)]

    cmd = str.join(' ', args)
    print('Executing %s' % cmd)
    subprocess.call(args)       
    
    if save_filenames != None:
        for ifile in range(len(save_filenames)):
            print save_filenames[ifile]
            flow_fn = 'pwc-net-pred-' + str(ifile).zfill(7) + '.flo'
            if not os.path.exists(flow_fn): flow_fn = os.path.join('./tmp', flow_fn)
            # to output the visualization, please download flow_io.py and viz_flow.py from https://github.com/jswulff/pcaflow/tree/master/pcaflow/utils
            # uv  = flow_read(flow_fn)
            # I_flow = Image.fromarray(viz_flow(uv[0], uv[1]))
            # imsave(vis_save_filenames[ifile], I_flow)    
            os.system ("mv  %s %s" % (flow_fn, save_filenames[ifile]) )

###################
my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir)

if not os.path.exists('./tmp'): os.mkdir('./tmp')

template        = './model/pwc_net_test.prototxt'
model_filename  = './model/pwc_net.caffemodel'   


image1_list     = './img1.txt';
image2_list     = './img2.txt';
out_flow_file   = './out.txt';
out_vis_file    = None;

if len(sys.argv) > 1:
    image1_list = sys.argv[1]
if len(sys.argv) > 2:
    image2_list = sys.argv[2]
if len(sys.argv) > 3:
    out_flow_file = sys.argv[3]
if len(sys.argv) > 4:
    out_vis_file = sys.argv[4]

with open(out_flow_file, 'r') as f:
    out_flow_list = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
if out_vis_file != None:
    with open(out_vis_file, 'r') as f:
        out_vis_list = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
else:
    out_vis_list = None

# for small images, it better set the scale_ratio to be 2.0 or 3.0 so that the input has height/width around 1000
scale_ratio = 1.0
evaluate_model(template, model_filename, image1_list, image2_list, out_flow_list, out_vis_list, scale_ratio)

