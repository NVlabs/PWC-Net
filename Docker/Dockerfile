FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
    rsync \
    htop \
    git \
    openssh-server \
    nano \
    cmake \
    python-opencv \
    python2.7 \
    unzip \
    time


RUN apt-get -qq -y install curl bzip2 && \
    curl -sSL https://repo.continuum.io/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh && \
    conda install -y python=2 && \
    conda update conda && \
    apt-get -qq -y autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
    conda clean --all --yes
    

# install external packages 
ADD torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
RUN pip install torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
RUN pip install torchvision==0.2.1 visdom==0.1.8.8 dominate==2.3.5 opencv-python==4.1.0.25 cffi==1.12.2

ADD external_packages external_packages

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

