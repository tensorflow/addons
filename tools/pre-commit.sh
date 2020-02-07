#!/usr/bin/env bash

set -e

DOCKER_BUILDKIT=1 docker build -t tf_addons_formatting -f tools/docker/Dockerfile_formatting .
docker run --rm -t -v "$(pwd -P):/addons" tf_addons_formatting
