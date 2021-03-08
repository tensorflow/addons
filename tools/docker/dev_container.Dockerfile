#syntax=docker/dockerfile:1.1.5-experimental
ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as base

ARG TF_PACKAGE
ARG TF_VERSION

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
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
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 ADDONS_DEV_CONTAINER=1

# Temporary until custom-op container is updated
RUN ln -sf $(which python3) /usr/bin/python
RUN ln -sf $(which pip3) /usr/local/bin/pip

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