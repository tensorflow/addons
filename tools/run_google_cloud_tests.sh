set -x -e

docker info
bash tools/run_gpu_tests.sh
