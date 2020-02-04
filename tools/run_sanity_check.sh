set -e
DOCKER_BUILDKIT=0 docker build -f tools/docker/sanity_check.Dockerfile ./
