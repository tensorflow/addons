set -e

docker buildx build \
    -f tools/docker/sanity_check.Dockerfile \
    --cache-from=type=registry,ref=gabrieldemarmiesse/caching-stuff \
    --target=${1} \
    ./
