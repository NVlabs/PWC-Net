#!/usr/bin/env python

from __future__ import print_function
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import subprocess
from pwc_net_utils import *


def make_net():
    # to modify to your local directory
    lmdb_file  = '../data/dispflownet-release/data/FlyingChairs_release_lmdb'
    split_list = '../data/dispflownet-release/data/FlyingChairs_release_test_train_split.list'        
   
    net_filename = './train.prototxt';
    with open(net_filename, 'w') as f:
        print(make_net_train(lmdb_file, split_list, 8), file=f)

    net_filename = './test.txt';
    with open(net_filename, 'w') as f:
        print(make_net_test(), file=f)
    # delete first 18 lines (unused image reading layer)        
    lines = open(net_filename).readlines()
    open(net_filename, 'w').writelines(lines[18:-1]);

    subprocess.call(["cat ./test_start.prototxt ./test.txt ./test_end.prototxt >./test.prototxt"], shell=True)

if __name__ == '__main__':
    make_net()