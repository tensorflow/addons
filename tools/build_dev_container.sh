#!/usr/bin/env bash

set -x -e

docker build \
    -f tools/docker/dev_container.Dockerfile \
    --build-arg TF_VERSION=2.12.0 \
    --build-arg TF_PACKAGE=tensorflow \
    --build-arg PY_VERSION=$PY_VERSION \
    --no-cache \
    --target dev_container \
    -t tfaddons/dev_container:latest-gpu ./
