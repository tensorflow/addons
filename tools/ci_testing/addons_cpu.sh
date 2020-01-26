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
set -x -e

# Make sure we're in the project root path.
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
ROOT_DIR=$( cd "$SCRIPT_DIR/../.." && pwd -P )
cd $ROOT_DIR
if [[ ! -d "tensorflow_addons" ]]; then
    echo "ERROR: PWD: $PWD is not project root"
    exit 1
fi

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

if [[ ${PLATFORM} == "darwin" ]]; then
    N_JOBS=$(sysctl -n hw.ncpu)
else
    N_JOBS=$(grep -c ^processor /proc/cpuinfo)
fi

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

export CC_OPT_FLAGS='-mavx'
export TF_NEED_CUDA=0

# Use default configuration here.
echo 'y' | ./configure.sh

## Run bazel test command. Double test timeouts to avoid flakes.
${BAZEL_PATH:=bazel} test -c opt -k \
    --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 \
    --test_output=errors --local_test_jobs=8 \
    --extra_toolchains=@bazel_tools//tools/python:autodetecting_toolchain_nonstrict \
    //tensorflow_addons/...


# running all the tests in tests/
${BAZEL_PATH:=bazel} build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl

# we need to move in the directory to avoid import issues
cd tests/tensorflow_addons && python -m unittest discover -s ./ -p '*_test.py'
