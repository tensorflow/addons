#!/usr/bin/env bash
# usage: bash tools/pre-commit.sh


set -e

if [ -z "${ADDONS_DEV_CONTAINER}" ]; then
  export DOCKER_BUILDKIT=1
  docker build -t tf_addons_formatting -f tools/docker/pre-commit.Dockerfile .

  export MSYS_NO_PATHCONV=1
  docker run --rm -t -v "$(pwd -P):/addons" tf_addons_formatting
else
  python tools/format.py
fi
