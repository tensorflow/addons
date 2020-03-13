# usage: bash tools/run_sanity_check.sh
# by default uses docker buildkit.
# to disable it:
# DOCKER_BUILDKIT=0 bash tools/run_sanity_check.sh
set -e

if [ "$DOCKER_BUILDKIT" == "" ]; then
  export DOCKER_BUILDKIT=1
fi

docker build -f tools/docker/sanity_check.Dockerfile ./
