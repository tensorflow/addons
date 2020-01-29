#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
# Make sure we're in the project root path.

# Optional environment variables for testing with multiple GPUs at once
# export TF_GPU_COUNT=4 # Specify number of GPUs available
# export TF_TESTS_PER_GPU=8 # Specify number of tests per GPU
# export TF_PER_DEVICE_MEMORY_LIMIT_MB=1024 # Limit the memory used per test
set -x

SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
ROOT_DIR=$( cd "$SCRIPT_DIR/../.." && pwd -P )
cd $ROOT_DIR
if [[ ! -d "tensorflow_addons" ]]; then
    echo "ERROR: PWD: $PWD is not project root"
    exit 1
fi

N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

export CC_OPT_FLAGS='-mavx'
export TF_NEED_CUDA="1"
export TF_CUDA_VERSION="10.1"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export TF_CUDNN_VERSION="7"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

# Check if python3 is available. On Windows it is not.
if [ -x "$(command -v python3)" ]; then
    echo 'y' | python3 ./configure.py
  else
    echo 'y' | python ./configure.py
fi

## Run bazel test command. Double test timeouts to avoid flakes.
bazel test -c opt -k \
    --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 \
    --test_output=errors --local_test_jobs=8 \
    --run_under=$(readlink -f tools/ci_testing/parallel_gpu_execute.sh) \
    --crosstool_top=//build_deps/toolchains/gcc7_manylinux2010-nvcc-cuda10.1:toolchain \
    --extra_toolchains=@bazel_tools//tools/python:autodetecting_toolchain_nonstrict \
    //tensorflow_addons/...

exit $?
