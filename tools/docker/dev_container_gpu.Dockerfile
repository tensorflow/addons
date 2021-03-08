#syntax=docker/dockerfile:1.1.5-experimental
ARG UBUNTU_VERSION=18.04

ARG ARCH=
ARG CUDA=11.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.0.4.30-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG TF_PACKAGE
ARG TF_VERSION

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        libcublas-dev-${CUDA/./-} \
        cuda-cudart-dev-${CUDA/./-} \
        libcufft-dev-${CUDA/./-} \
        libcurand-dev-${CUDA/./-} \
        libcusolver-dev-${CUDA/./-} \
        libcusparse-dev-${CUDA/./-} \
        libcudnn8=${CUDNN}+cuda${CUDA} \
        libcudnn8-dev=${CUDNN}+cuda${CUDA} \
        libcurl3-dev \
        libfreetype6-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        git \
        swig \
        curl \
        python3 \
        python3-pip \
        python3-dev \
        && \
    find /usr/local/cuda-${CUDA}/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/${LIB_DIR_PREFIX}-linux-gnu/libcudnn_static_v8.a && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/include/x86_64-linux-gnu:$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs \
    TF_NEED_CUDA=1 TF_CUDA_VERSION=${CUDA} TF_CUDNN_VERSION=${CUDNN_MAJOR_VERSION} LANG=C.UTF-8 ADDONS_DEV_CONTAINER=1

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

RUN ln -sf $(which python3) /usr/bin/python && ln -sf $(which pip3) /usr/local/bin/pip

COPY tools/install_deps /install_deps
COPY requirements.txt /tmp/requirements.txt

RUN pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

RUN pip --no-cache-dir install -r /install_deps/black.txt \
    -r /install_deps/flake8.txt \
    -r /install_deps/pytest.txt \
    -r /install_deps/typedapi.txt \
    -r /tmp/requirements.txt

RUN pip --no-cache-dir install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh
RUN bash /install_deps/install_bazelisk.sh