#!/usr/bin/env bash

set -x -e

docker build \
    -f tools/docker/dev_container.Dockerfile \
    --build-arg TF_VERSION=2.7.0 \
    --build-arg TF_PACKAGE=tensorflow-cpu \
    --build-arg PY_VERSION=$PY_VERSION \
    --no-cache \
    --target dev_container_cpu \
    -t tfaddons/dev_container:latest-cpu ./
