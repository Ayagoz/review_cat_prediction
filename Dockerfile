FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV PYTHONPATH $PYTHONPATH:/workdir
WORKDIR /workdir

# Install Python and apt-get packages
RUN apt-get update && apt -y upgrade && \
    apt-get -y install software-properties-common apt-utils && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get -y install \
    build-essential yasm nasm ninja-build cmake \
    unzip git wget curl nano vim tmux htop nvtop \
    sysstat libtcmalloc-minimal4 pkgconf \
    autoconf libtool flex bison \
    libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    libx264-dev libsndfile1 libssl-dev \
    python3.10 python3.10-dev python3.10-distutils python3.10-tk \
    libpng-dev libjpeg-dev libmp3lame-dev \
    liblapack-dev libopenblas-dev gfortran && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives/*

# Install pip
RUN PYTHON_PIP_VERSION=22.2.2 && \
    PYTHON_SETUPTOOLS_VERSION=49.1.2 && \
    wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py && \
	export PYTHONDONTWRITEBYTECODE=1 && \
	python get-pip.py \
		--disable-pip-version-check \
		--no-cache-dir \
		--no-compile \
		"pip==$PYTHON_PIP_VERSION" \
		"setuptools==$PYTHON_SETUPTOOLS_VERSION" && \
	rm -f get-pip.py && \
	pip --version

WORKDIR /workdir

COPY requirements.txt /workdir
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /workdir
COPY Makefile /workdir
COPY Dockerfile /workdir