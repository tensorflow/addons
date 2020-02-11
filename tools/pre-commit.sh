#!/usr/bin/env bash

set -e

export DOCKER_BUILDKIT=1
docker build -t tf_addons_formatting -f tools/docker/pre-commit.Dockerfile .
docker run --rm -t -v "$(pwd -P):/addons" tf_addons_formatting
