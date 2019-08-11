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
# Usage: configure.sh [--quiet]
#
# Options:
#  --quiet  Give less output.


# Writes variables to bazelrc file
function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Converts the linkflag namespec to the full shared library name
function generate_shared_lib_name() {
  if [[ $(uname) == "Darwin" ]]; then
    local namespec="$1"
    echo "lib"${namespec:2}".dylib"
  else
    local namespec="$1"
    echo ${namespec:3}
  fi
}

QUIET_FLAG=""
if [[ $1 == "--quiet" ]]; then
    QUIET_FLAG="--quiet"
elif [[ ! -z "$1" ]]; then
    echo "Found unsupported args: $@"
    exit 1
fi

# Install python dependencies
read -r -p "Tensorflow 2.0 will be installed if it is not already. Are You Sure? [y/n] " reply
case $reply in
    [yY]*) echo "Installing...";;
    * ) echo "Goodbye!"; exit;;
esac

BUILD_DEPS_DIR=build_deps
REQUIREMENTS_TXT=$BUILD_DEPS_DIR/requirements.txt
if [[ "$TF_NEED_CUDA" == "1" ]]; then
    # TODO: delete it when tf2 standard package supports
    # both cpu and gpu kernel.
    REQUIREMENTS_TXT=$BUILD_DEPS_DIR/requirements_gpu.txt
fi

${PYTHON_VERSION:=python} -m pip install $QUIET_FLAG -r $REQUIREMENTS_TXT

[[ -f .bazelrc ]] && rm .bazelrc

TF_CFLAGS=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CXX11_ABI_FLAG=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)') )

TF_SHARED_LIBRARY_DIR=${TF_LFLAGS[0]:2}
TF_SHARED_LIBRARY_NAME=$(generate_shared_lib_name ${TF_LFLAGS[1]})

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${TF_SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_CXX11_ABI_FLAG}

write_to_bazelrc "build:manylinux2010 --crosstool_top=//build_deps/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain"
write_to_bazelrc "build --config=manylinux2010"
write_to_bazelrc "test --config=manylinux2010"

if [[ "$TF_NEED_CUDA" == "1" ]]; then
    write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "/usr/lib/x86_64-linux-gnu"
    write_action_env_to_bazelrc "TF_CUDA_VERSION" "10.0"
    write_action_env_to_bazelrc "TF_CUDNN_VERSION" "7"
    write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "${CUDA_HOME:=/usr/local/cuda}"
    write_to_bazelrc "build --config=cuda"
    write_to_bazelrc "test --config=cuda"

    write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
    write_to_bazelrc "build --spawn_strategy=standalone"
    write_to_bazelrc "build --strategy=Genrule=standalone"
    write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}

fi
