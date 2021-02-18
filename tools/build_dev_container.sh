#!/usr/bin/env bash

set -x -e

DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/dev_container.Dockerfile \
    --build-arg TF_VERSION=2.4.1 \
    --build-arg TF_PACKAGE=tensorflow-cpu \
    --target dev_container_cpu \
    -t tfaddons/dev_container:latest-cpu ./
