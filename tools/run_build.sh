set -e

export DOCKER_BUILDKIT=1
docker build -f tools/docker/sanity_check.Dockerfile --target=${1} ./
