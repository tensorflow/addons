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
# usage: bash tools/ci_testing/addons_cpu.sh [--no-deps]

set -x -e


if [ "$1" != "--no-deps" ] && [ "$1" != "" ]; then
  echo Wrong argument $1
  exit 1
fi

# Make sure we're in the project root path.
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
ROOT_DIR=$( cd "$SCRIPT_DIR/../.." && pwd -P )
cd $ROOT_DIR
if [[ ! -d "tensorflow_addons" ]]; then
    echo "ERROR: PWD: $PWD is not project root"
    exit 1
fi

export CC_OPT_FLAGS='-mavx'
export TF_NEED_CUDA=0

# Check if python3 is available. On Windows VM it is not.
if [ -x "$(command -v python3)" ]; then
    python3 -m pip install -r tools/tests_dependencies/pytest.txt
    python3 ./configure.py $1
    cat ./.bazelrc
    bash tools/install_so_files.sh
    python3 -m pytest --cov=tensorflow_addons -v --durations=25 -n auto ./tensorflow_addons
  else
    python -m pip install -r tools/tests_dependencies/pytest.txt
    python ./configure.py $1
    cat ./.bazelrc
    bash tools/install_so_files.sh
    python -m pytest --cov=tensorflow_addons -v --durations=25 -n auto ./tensorflow_addons
fi


exit $?
