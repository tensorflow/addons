# usage: bash tools/run_gpu_tests.sh

set -x -e

export DOCKER_BUILDKIT=1
docker build \
       -f tools/docker/build_wheel.Dockerfile \
       --target tfa_gpu_tests \
       --build-arg TF_VERSION=2.1.0 \
       --build-arg PY_VERSION=3.5 \
       -v cache_bazel:/root/.cache/bazel \
       -t tfa_gpu_tests ./
docker run --rm -t --runtime=nvidia tfa_gpu_tests
