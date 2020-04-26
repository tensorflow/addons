# usage: bash tools/run_sanity_check.sh

set -e

export DOCKER_BUILDKIT=1
docker build -f tools/docker/sanity_check.Dockerfile ./
