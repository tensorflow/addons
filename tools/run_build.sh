# usage: bash tools/run_build.sh
# by default uses docker buildkit.
# to disable it:
# DOCKER_BUILDKIT=0 bash tools/run_build.sh
set -e

export DOCKER_BUILDKIT=1
docker build -f tools/docker/sanity_check.Dockerfile --target=${1} ./
