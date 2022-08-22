# Running PWC-Net on Docker and working with a remote interpreter in PyCharm on Linux:

### Worked for me on the following system:

Ubuntu 20.04.4 LTS

docker 20.10.17, build 100c701

nvidia-docker 2.11.0-1

GPU: GeForce RTX 2080 Ti

 
### Full installation and and settings configuration instructions:

Step 1: Look at [NVIDIA official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for installing `nvdia-docker2` (which include instructions for installing docker)

make sure [step 3](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#step-3-testing-the-installation) of it runs as expected. 

Step 2: [Download](https://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl) the appropriate torch version for this project.

Step 3: Put the `torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl`, the Dockerfile and the external packages folder in the same folder anywhere you like.

Step 4: got to the folder where your Dockerfile located through the terminal and run `docker build -t choose_any_tag_name .`

Now you should be able to see your docker img tag in Docker in your PyCharm services:

![image](https://user-images.githubusercontent.com/50303550/185483233-b77c45b1-75a9-404d-8fdb-c4a7aad2f72f.png)

Step 5: As another sainity check, `Create Container` (right mouse click on the img tag) with the following run options arguments `--gpus=all --runtime=nvidia` (click on `Modify Options`, then on `Run Options`):

![image](https://user-images.githubusercontent.com/50303550/185483940-f272742a-b15d-44bc-aa1d-a222b9d1e433.png)

You may set the container name and the Docker name as you desire.


Step 6: Create a project in PyCharm and put all the files from the repository in there except the `external packages`, which you should take from this folder.

Step 7: in file `models/PWC-Net.py` at the head of it, add:

`import os ;
os.system("bash /external_packages/correlation-pytorch-master/make_cuda.sh")
`

Step 7: Go to Python Interpreter window in PyCharm and click on `Add Interpreter` and choose `On Docker`:

![image](https://user-images.githubusercontent.com/50303550/185485149-87092a37-5d2c-48c1-a4bc-57c63412666f.png)


 choose option `pull` and write your image tag and then add `:latest` (it should pop up from the list as you can see below) and click `next`:

![image](https://user-images.githubusercontent.com/50303550/185485744-6e3b9d5b-ff78-4c78-9c4a-c35b90e4d3af.png)

it should end with `Introspection completed ; Removing introspection container`. click again `next`.


choose `System Enterpreter`, then click on the `...` and enter `/usr/local/bin/python` (this should be the path on your docker to the python interpreter).
Notice that you didn't choose it yet, just click on the toggle arrow and click on the path you've just added.


### You are done and you should see how the interpreter now updates in libraries and in a couple of minutes it is ready for running and debugging. 

If everything worked, the first thing you would see each time your run the console is how the external packages are compiled with nvcc and the correlation layer is installed. 
Notice this must happen each run because docker running a clean version of the container with everything installed from your build command in step 4.

For compiling the external_packages on the docker in file `make_cuda.sh` I changed the setup command to `pip install .` and also set the absolute path from `~` directory on the docker.
also in `external_packages/correlation-pytorch-master/correlation-pytorch/correlation_package/_ext/corr/__init__.py` 

I modificated the import to be:

`from torch.utils.ffi import _wrap_function ;
import imp ;
_corr = imp.load_dynamic('_corr','/usr/local/lib/python2.7/site-packages/correlation_package/_ext/corr/_corr.so') ;
from _corr import lib as _lib, ffi as _ffi` ;

eventually that how it worked for me. 


# Super important note:

notice the authors used a GPU of Pascal series for which CUDNN of particular versions only suitable and there for CUDA and torch.
My GPU wasn't compatible, because for torch 0.2.0, CUDA8.0 and compatibele CUDNN a GPU with compute capability of 6.x is needed.
But some how, except very slow few first GPU commands such as `.cuda()` it worked. 

If you want to be able to run it without such side-effects, you will probably have to run on it on some older GPU with compute capability of 6.x, 
and the sure solution is to run it with a similiar GPU version to which the authors mention in the paper. 



















