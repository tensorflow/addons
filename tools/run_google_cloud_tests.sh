set -x -e


DOCKER_BUILDKIT=1 bash tools/run_gpu_tests.sh
