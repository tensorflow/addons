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
# usage: bash tools/testing/build_and_run_tests.sh

set -x -e

SKIP_CUSTOM_OP_TESTS_FLAG=${1}

python -m pip install -r tools/install_deps/pytest.txt -e ./
python ./configure.py
bash tools/install_so_files.sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"

# use 10 workers if a gpu is available, otherwise,
# one worker per cpu core. Kokoro has 38 cores, that'd be too much
# for the gpu memory, until we change the device placement to
# use multiple gpus when they are available.
EXTRA_ARGS="-n 10"
if ! [ -x "$(command -v nvidia-smi)" ]; then
  EXTRA_ARGS="-n auto"
fi

bazel clean
python -m pytest -v --functions-durations=20 --modules-durations=5 $SKIP_CUSTOM_OP_TESTS_FLAG $EXTRA_ARGS ./tensorflow_addons
