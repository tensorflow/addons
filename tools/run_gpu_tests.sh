# usage: bash tools/run_gpu_tests.sh

set -x -e

export DOCKER_BUILDKIT=1
docker build \
       -f tools/docker/build_wheel.Dockerfile \
       --target tfa_gpu_tests \
       --build-arg TF_VERSION=2.7.0rc0 \
       --build-arg PY_VERSION=3.6 \
       -t tfa_gpu_tests ./
docker run --rm -t --gpus=all tfa_gpu_tests
