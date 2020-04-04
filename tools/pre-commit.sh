#!/usr/bin/env bash
# usage: bash tools/pre-commit.sh
# by default uses docker buildkit.
# to disable it:
# DOCKER_BUILDKIT=0 bash tools/pre-commit.sh


set -e

export DOCKER_BUILDKIT=1
docker build -t tf_addons_formatting -f tools/docker/pre-commit.Dockerfile .

export MSYS_NO_PATHCONV=1
docker run --rm -t -v "$(pwd -P):/addons" tf_addons_formatting
