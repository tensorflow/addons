#!/usr/bin/env bash

set -x -e

docker build -t tf_addons_formatting -f tools/docker/Dockerfile_formatting .
docker run --rm -t -v "$(pwd -P):/addons" tf_addons_formatting
