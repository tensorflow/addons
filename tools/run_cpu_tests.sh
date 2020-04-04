# usage: bash tools/run_cpu_tests.sh

set -e

export DOCKER_BUILDKIT=1
docker build --progress=plain -f tools/docker/cpu_tests.Dockerfile ./
