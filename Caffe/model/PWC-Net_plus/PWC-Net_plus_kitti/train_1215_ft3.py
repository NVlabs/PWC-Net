#!/usr/bin/env python
import os, sys
import subprocess

caffe_bin = '../../../build/tools/caffe.bin'


os.system('mkdir training_ft3') 
os.chdir('training_ft3') 

# =========================================================

my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir)


if os.path.exists('./core'):
	os.remove('./core')

if not os.path.isfile(caffe_bin):
    print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
    sys.exit(1)

print('args:', sys.argv[1:])

trained_filenames = os.listdir('./')

if len(trained_filenames)==0:
	args = [caffe_bin, 'train', '-solver', '../model/solver_1215_ft3.prototxt', '-weights', '../training_ft2/flow_iter_150000.caffemodel'] + sys.argv[1:]	
else:
	# start from the latest training result
	iters = []
	for i in range(len(trained_filenames)):
		i0 = trained_filenames[i].find('iter_')
		if  i0==-1:
			continue
		i1 = trained_filenames[i].find('.')
		iters.append(int(trained_filenames[i][i0+5:i1]))		
	latest_iter = max(iters)
	args = [caffe_bin, 'train', '-solver', '../model/solver_1215_ft3.prototxt', '-snapshot', 'flow_iter_'+ str(latest_iter) + '.solverstate'] + sys.argv[1:]
	
cmd = str.join(' ', args)
print('Executing %s' % cmd)

subprocess.call(args)
