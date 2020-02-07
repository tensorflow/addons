set -e

export DOCKER_CLI_EXPERIMENTAL=enabled
docker buildx build \
    -f tools/docker/sanity_check.Dockerfile \
    --cache-from=type=registry,ref=gabrieldemarmiesse/caching-stuff \
    --target=${1} \
    ./
