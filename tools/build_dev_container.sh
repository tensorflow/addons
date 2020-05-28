#!/usr/bin/env bash

set -x -e

DOCKER_BUILDKIT=1 docker build \
    --cache-from tfaddons/dev_container:latest \
    -f tools/docker/build_wheel.Dockerfile \
    --target base_install \
    --build-arg TF_VERSION=2.2.0 \
    --build-arg PY_VERSION=3.6 \
    -t addons_base:latest ./

DOCKER_BUILDKIT=1 docker build \
    --cache-from tfaddons/dev_container:latest \
    -f tools/docker/dev_container.Dockerfile \
    --target build_dev_container \
    -t tfaddons/dev_container:latest ./
