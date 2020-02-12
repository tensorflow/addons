# usage: bash tools/run_gpu_tests.sh [--no-deps]

set -x -e

if [ "$1" != "--no-buildkit" ] && [ "$1" != "" ]; then
  echo Wrong argument $1
  exit 1
fi

if [ "$1" == "--no-buildkit" ]; then
  export DOCKER_BUILDKIT=0
else
  export DOCKER_BUILDKIT=1
fi

docker build -f tools/docker/gpu_tests.Dockerfile -t tfa_gpu_tests ./
docker run --rm -t --runtime=nvidia tfa_gpu_tests
