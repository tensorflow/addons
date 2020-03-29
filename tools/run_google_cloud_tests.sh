set -x -e

docker info
DOCKER_BUILDKIT=0 bash tools/run_gpu_tests.sh
