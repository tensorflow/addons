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

export CC_OPT_FLAGS='-mavx'

python -m pip install -r tools/install_deps/pytest.txt -e ./
python ./configure.py
bash tools/install_so_files.sh
nvidia-smi
echo logical devices
python -c "import tensorflow as tf; print(tf.config.list_logical_devices())"
echo gpu available
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
python -m pytest -v --durations=25 ./tensorflow_addons
