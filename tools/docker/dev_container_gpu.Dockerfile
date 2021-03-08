ARG UBUNTU_VERSION=18.04

ARG ARCH=
ARG CUDA=11.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
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
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/include/x86_64-linux-gnu:$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs \
    TF_NEED_CUDA=1 TF_CUDA_VERSION=${CUDA} LANG=C.UTF-8 ADDONS_DEV_CONTAINER="1"

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# Temporary until custom-op container is updated
RUN ln -sf $(which python3) /usr/bin/python
RUN ln -sf $(which pip3) /usr/local/bin/pip

COPY tools/install_deps /install_deps
COPY requirements.txt /tmp/requirements.txt

RUN https_proxy=http://10.15.5.156:1083 http_proxy=http://10.15.5.156:1083 pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools \
    -r /install_deps/black.txt \
    -r /install_deps/flake8.txt \
    -r /install_deps/pytest.txt \
    -r /install_deps/typedapi.txt \
    -r /tmp/requirements.txt

RUN https_proxy=http://10.15.5.156:1083 http_proxy=http://10.15.5.156:1083 pip install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

RUN https_proxy=http://10.15.5.156:1083 http_proxy=http://10.15.5.156:1083 bash /install_deps/buildifier.sh
RUN https_proxy=http://10.15.5.156:1083 http_proxy=http://10.15.5.156:1083 bash /install_deps/clang-format.sh
RUN https_proxy=http://10.15.5.156:1083 http_proxy=http://10.15.5.156:1083 bash /install_deps/install_bazelisk.sh