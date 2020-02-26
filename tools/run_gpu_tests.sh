# usage: bash tools/run_gpu_tests.sh
# by default uses docker buildkit.
# to disable it:
# DOCKER_BUILDKIT=0 bash tools/run_gpu_tests.sh

set -x -e

if [ "$DOCKER_BUILDKIT" == "" ]; then
  export DOCKER_BUILDKIT=1
fi

docker build -f tools/docker/gpu_tests.Dockerfile -t tfa_gpu_tests ./
docker run --rm -t --runtime=nvidia tfa_gpu_tests
