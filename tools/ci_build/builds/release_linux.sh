#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

if [[ $2 == "--nightly" ]]; then
    PKG_OPS="--nightly"
elif [[ -n "$2" ]]; then
    echo "Found unsupported args: $@"
    exit 1
fi

# Configs
export TF_NEED_CUDA="1"
export TF_CUDA_VERSION="10.1"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export TF_CUDNN_VERSION="7"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

# Fix presented in
# https://stackoverflow.com/questions/44967202/pip-is-showing-error-lsb-release-a-returned-non-zero-exit-status-1/44967506
echo "#! /usr/bin/python2.7" >> /usr/bin/lsb_release2
cat /usr/bin/lsb_release >> /usr/bin/lsb_release2
mv /usr/bin/lsb_release2 /usr/bin/lsb_release

ln -sf $(which python$1) /usr/bin/python3
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools

#Link TF dependency
python3 --version
python3 ./configure.py

bash tools/ci_testing/addons_cpu.sh

# Build
bazel build \
  -c opt \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  --crosstool_top=//build_deps/toolchains/gcc7_manylinux2010-nvcc-cuda10.1:toolchain \
  build_pip_pkg

# Package Whl
bazel-bin/build_pip_pkg artifacts ${PKG_OPS}
