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

if [[ $1 == "--nightly" ]]; then
    PKG_OPS="--nightly"
elif [[ -n "$1" ]]; then
    echo "Found unsupported args: $@"
    exit 1
fi

# Configs
export TF_NEED_CUDA="1"
export TF_CUDA_VERSION="10.1"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export TF_CUDNN_VERSION="7"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

# Remove the now private ppa. This can be removed after the docker image removes the
# pre-installed python packages from this ppa.
rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list

PYTHON_VERSIONS="python3.5 python3.6 python3.7"
ln -sf /usr/bin/python3.5 /usr/bin/python3 # Py36 has issues with add-apt
curl -sSOL https://bootstrap.pypa.io/get-pip.py
add-apt-repository -y ppa:deadsnakes/ppa

apt-get -y -qq update

for version in ${PYTHON_VERSIONS}; do
    apt-get -y -qq install ${version}
    ln -sf /usr/bin/${version} /usr/bin/python

    python get-pip.py -q
    python -m pip --version

    #Link TF dependency
    yes 'y' | ./configure.sh --quiet

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
done

# Clean up
rm get-pip.py
