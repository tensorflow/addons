set -e -x

df -h
docker info
rm -rf /usr/share/dotnet &
df -h

DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/build_wheel.Dockerfile \
    --output type=local,dest=wheelhouse \
    --build-arg PY_VERSION \
    --build-arg TF_VERSION \
    --build-arg NIGHTLY_FLAG \
    --build-arg NIGHTLY_TIME \
    ./
