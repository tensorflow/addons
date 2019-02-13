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

set -e
set -x

N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

export CC_OPT_FLAGS='-mavx'
export TF_NEED_CUDA=0 # TODO: Verify this be used in GPU custom-op?

######### Python3 Tests
# TODO: tf-nightly-2.0-preview is unavailable for python3.4


#export PYTHON_BIN_PATH=`which python3`
#/bin/bash configure.sh --py_version=3

## Run bazel test command. Double test timeouts to avoid flakes.
#bazel test -c opt -k \
#    --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 \
#    --test_output=errors --local_test_jobs=8 \
#    //tensorflow_addons/...


######### Python2 Tests
export PYTHON_BIN_PATH=`which python2`
/bin/bash configure.sh --py_version=2

## Run bazel test command. Double test timeouts to avoid flakes.
bazel test -c opt -k \
    --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 \
    --test_output=errors --local_test_jobs=8 \
    //tensorflow_addons/...