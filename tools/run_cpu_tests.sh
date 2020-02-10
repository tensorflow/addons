set -e

export DOCKER_BUILDKIT=1
docker build -f tools/docker/cpu_tests.Dockerfile ./
