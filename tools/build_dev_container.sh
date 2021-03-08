#!/usr/bin/env bash

set -x -e

TF_VERSION=2.4.1

DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/dev_container.Dockerfile \
    --build-arg TF_VERSION=$TF_VERSION \
    --build-arg TF_PACKAGE=tensorflow-cpu \
    -t tfaddons/dev_container:latest-cpu ./

DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/dev_container_gpu.Dockerfile \
    --build-arg TF_VERSION=$TF_VERSION \
    --build-arg TF_PACKAGE=tensorflow \
    --build-arg CUDA=11.0 \
    -t tfaddons/dev_container:latest-gpu ./
