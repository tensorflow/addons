# usage: bash tools/run_cpu_tests.sh
# by default uses docker buildkit.
# to disable it:
# DOCKER_BUILDKIT=0 bash tools/run_cpu_tests.sh
set -e

if [ "$DOCKER_BUILDKIT" == "" ]; then
  export DOCKER_BUILDKIT=1
fi

docker build -f tools/docker/cpu_tests.Dockerfile ./
