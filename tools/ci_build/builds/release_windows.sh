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
# ==============================================================================
set -e -x

# No GPU support for Windows (See #784)
export TF_NEED_CUDA="0"

mkdir -p artifacts/
export BAZEL_VC=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/VC/

# Install Bazel 1.1.0
wget --quiet -nc https://github.com/bazelbuild/bazel/releases/download/1.1.0/bazel-1.1.0-windows-x86_64.exe

python --version
python -m pip install --upgrade pip

#Link TF dependency
echo 'y' | ./configure.sh --quiet

./bazel-1.1.0-windows-x86_64.exe build \
    -c opt \
    --enable_runfiles \
    --noshow_progress \
    --noshow_loading_progress \
    --verbose_failures \
    --test_output=errors \
    build_pip_pkg

bazel-bin/build_pip_pkg artifacts --nightly
