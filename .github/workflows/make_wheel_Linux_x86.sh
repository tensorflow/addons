set -e -x

df -h
docker info

# to get more disk space
rm -rf "$AGENT_TOOLSDIRECTORY" &
rm -rf /opt/ghc &
rm -rf "/usr/local/share/boost" &
rm -rf /usr/share/dotnet

# Tests are ran as part of make_wheel target
DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/build_wheel.Dockerfile \
    --output type=local,dest=wheelhouse \
    --build-arg PY_VERSION \
    --build-arg TF_VERSION \
    --build-arg NIGHTLY_FLAG \
    --build-arg NIGHTLY_TIME \
    --build-arg SKIP_CUSTOM_OP_TESTS \
    ./
