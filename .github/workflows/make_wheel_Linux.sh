set -e

df -h
docker info
# to get more disk space
rm -rf /usr/share/dotnet &

export DOCKER_CLI_EXPERIMENTAL=enabled
docker buildx create --driver docker-container --use

docker login -u $DOCKER_LOGIN -p $DOCKER_PASSWORD
docker buildx build \
    -f tools/docker/build_wheel.Dockerfile \
    --output type=local,dest=wheelhouse \
    --cache-to=type=registry,ref=gabrieldemarmiesse/my_pretty_cache,mode=max \
    --build-arg PY_VERSION \
    --build-arg TF_VERSION \
    --build-arg NIGHTLY_FLAG \
    --build-arg NIGHTLY_TIME \
    ./
