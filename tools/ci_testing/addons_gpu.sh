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
set -e -x

export CC_OPT_FLAGS='-mavx'
export TF_NEED_CUDA="1"
export TF_CUDA_VERSION="10.1"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export TF_CUDNN_VERSION="7"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

python3 -m pip install -r tools/tests_dependencies/pytest.txt
python3 ./configure.py
cat ./.bazelrc
bash tools/install_so_files.sh
python3 -m pytest --cov=tensorflow_addons -v --durations=25 -n auto ./tensorflow_addons
