# usage: bash tools/run_gpu_tests.sh

set -x -e

export DOCKER_BUILDKIT=1
docker build -f tools/docker/gpu_tests.Dockerfile -t tfa_gpu_tests ./
docker run --rm -t --runtime=nvidia tfa_gpu_tests
