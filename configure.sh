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


PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"


# Writes variables to bazelrc file
function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

function is_linux() {
  [[ "${PLATFORM}" == "linux" ]]
}

function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function is_ppc64le() {
  [[ "$(uname -m)" == "ppc64le" ]]
}

# Converts the linkflag namespec to the full shared library name
function generate_shared_lib_name() {
  if is_macos; then
    local namespec="$1"
    echo "lib"${namespec:2}".dylib"
  elif is_windows; then
    echo "_pywrap_tensorflow_internal.lib"
  else
    local namespec="$1"
    echo ${namespec:3}
  fi
}

PIP_INSTALL_OPTS="--upgrade"
if [[ $1 == "--quiet" ]]; then
    PIP_INSTALL_OPTS="$PIP_INSTALL_OPTS --quiet"
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

${PYTHON_VERSION:=python} -m pip install $PIP_INSTALL_OPTS -r $REQUIREMENTS_TXT

[[ -f .bazelrc ]] && rm .bazelrc

TF_CFLAGS=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CXX11_ABI_FLAG=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)') )

if is_windows; then
  # Use pywrap_tensorflow instead of tensorflow_framework on Windows
  TF_SHARED_LIBRARY_DIR=${TF_CFLAGS:2:-7}"python"
else
  TF_SHARED_LIBRARY_DIR=${TF_LFLAGS[0]:2}
fi

TF_SHARED_LIBRARY_NAME=$(generate_shared_lib_name ${TF_LFLAGS[1]})
TF_HEADER_DIR=${TF_CFLAGS:2}

if is_windows; then
  TF_SHARED_LIBRARY_DIR=${TF_SHARED_LIBRARY_DIR//\\//}
  TF_SHARED_LIBRARY_NAME=${TF_SHARED_LIBRARY_NAME//\\//}
  TF_HEADER_DIR=${TF_HEADER_DIR//\\//}
fi

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${TF_SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_CXX11_ABI_FLAG}

write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"

if [[ "$TF_NEED_CUDA" == "1" ]]; then
    write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}
    write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "${CUDNN_INSTALL_PATH:=/usr/lib/x86_64-linux-gnu}"
    write_action_env_to_bazelrc "TF_CUDA_VERSION" "10.1"
    write_action_env_to_bazelrc "TF_CUDNN_VERSION" "7"
    write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "${CUDA_HOME:=/usr/local/cuda}"

    write_to_bazelrc "test --config=cuda"
    write_to_bazelrc "build --config=cuda"
    write_to_bazelrc "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"
fi

echo "Finished donfigure...."