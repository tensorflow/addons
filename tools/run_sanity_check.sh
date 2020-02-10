set -e
DOCKER_BUILDKIT=1 docker build -f tools/docker/sanity_check.Dockerfile ./
