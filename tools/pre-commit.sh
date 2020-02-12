#!/usr/bin/env bash

set -e

if [ "$DOCKER_BUILDKIT" == "" ]; then
  export DOCKER_BUILDKIT=1
fi

docker build -t tf_addons_formatting -f tools/docker/pre-commit.Dockerfile .
docker run --rm -t -v "$(pwd -P):/addons" tf_addons_formatting
