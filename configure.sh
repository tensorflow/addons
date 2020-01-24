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
DEFAULT_CUDA_VERISON="10.1"
DEFAULT_CUDA_PATH="/usr/local/cuda"
DEFAULT_CUDNN_VERSION="7"
DEFAULT_CUDNN_PATH="/usr/lib/x86_64-linux-gnu"

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
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function is_ppc64le() {
  [[ "$(uname -m)" == "ppc64le" ]]
}

# Converts the linkflag namespec to the full shared library name
function generate_shared_lib_name() {
  if is_macos; then
    # MacOS
    local namespec="$1"
    echo "lib"${namespec:2}".dylib"
  elif is_windows; then
    # Windows
    echo "_pywrap_tensorflow_internal.lib"
  else
    # Linux
    local namespec="$1"
    echo ${namespec:3}
  fi
}

echo ""
echo "Configuring TensorFlow Addons to be built from source..."

PIP_INSTALL_OPTS="--upgrade"
if [[ $1 == "--quiet" ]]; then
    PIP_INSTALL_OPTS="$PIP_INSTALL_OPTS --quiet"
elif [[ -n "$1" ]]; then
    echo "Found unsupported args: $@"
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
PYTHON_PATH=$(which python)
REQUIRED_PKG=$(cat requirements.txt)

echo ""
echo "> TensorFlow Addons will link to the framework in a pre-installed TF pacakge..."
echo "> Checking installed packages in ${PYTHON_PATH}"
python build_deps/check_deps.py

if [[ $? == 1 ]]; then
  read -r -p "Package ${REQUIRED_PKG} will be installed. Are You Sure? [y/n] " reply
  case $reply in
      [yY]*) echo "> Installing..."
         python -m pip install $PIP_INSTALL_OPTS -r requirements.txt;;
      * ) echo "> Exiting..."; exit;;
  esac
else
  echo "> Using pre-installed ${REQUIRED_PKG}..."
fi

[[ -f .bazelrc ]] && rm .bazelrc

TF_CFLAGS=($(python -c 'import logging; logging.disable(logging.WARNING);import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import logging; logging.disable(logging.WARNING);import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
TF_CXX11_ABI_FLAG=($(python -c 'import logging; logging.disable(logging.WARNING);import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)'))

TF_SHARED_LIBRARY_NAME=$(generate_shared_lib_name ${TF_LFLAGS[1]})
TF_HEADER_DIR=${TF_CFLAGS:2}

# OS Specific parsing
if is_windows; then
  TF_SHARED_LIBRARY_DIR=${TF_CFLAGS:2:-7}"python"
  TF_SHARED_LIBRARY_DIR=${TF_SHARED_LIBRARY_DIR//\\//}

  TF_SHARED_LIBRARY_NAME=${TF_SHARED_LIBRARY_NAME//\\//}
  TF_HEADER_DIR=${TF_HEADER_DIR//\\//}
else
  TF_SHARED_LIBRARY_DIR=${TF_LFLAGS[0]:2}
fi

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${TF_SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_CXX11_ABI_FLAG}

write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"

while [[ "$TF_NEED_CUDA" == "" ]]; do
  echo ""
  read -p "Do you want to build GPU ops? [y/N] " INPUT
  case $INPUT in
    [Yy]* ) echo "> Building GPU & CPU ops"; TF_NEED_CUDA=1;;
    [Nn]* ) echo "> Building only CPU ops"; TF_NEED_CUDA=0;;
    "" ) echo "> Building only CPU ops"; TF_NEED_CUDA=0;;
    * ) echo "Invalid selection: " $INPUT;;
  esac
done

if [[ "$TF_NEED_CUDA" == "1" ]]; then
    echo ""
    echo "Configuring GPU setup..."

    while [[ "$TF_CUDA_VERSION" == "" ]]; do
      read -p "Please specify the CUDA version [Default is $DEFAULT_CUDA_VERISON]: " INPUT
      case $INPUT in
        "" ) echo "> Using CUDA version: 10.1"; TF_CUDA_VERSION=$DEFAULT_CUDA_VERISON;;
        * ) echo "> Using CUDA version:" $INPUT; TF_CUDA_VERSION=$INPUT;;
      esac
    echo ""
    done

    while [[ "$CUDA_TOOLKIT_PATH" == "" ]]; do
      read -p "Please specify the location of CUDA. [Default is $DEFAULT_CUDA_PATH]: " INPUT
      case $INPUT in
        "" ) echo "> CUDA installation path: /usr/local/cuda"; CUDA_TOOLKIT_PATH=$DEFAULT_CUDA_PATH;;
        * ) echo "> CUDA installation path:" $INPUT; CUDA_TOOLKIT_PATH=$INPUT;;
      esac
    echo ""
    done

    while [[ "$TF_CUDNN_VERSION" == "" ]]; do
      read -p "Please specify the cuDNN major version [Default is $DEFAULT_CUDNN_VERSION]: " INPUT
      case $INPUT in
        "" ) echo "> Using cuDNN version: 7"; TF_CUDNN_VERSION=$DEFAULT_CUDNN_VERSION;;
        * ) echo "> Using cuDNN version:" $INPUT; TF_CUDNN_VERSION=$INPUT;;
      esac
    echo ""
    done

    while [[ "$CUDNN_INSTALL_PATH" == "" ]]; do
      read -p "Please specify the location of cuDNN installation. [Default is $DEFAULT_CUDNN_PATH]: " INPUT
      case $INPUT in
        "" ) echo "> cuDNN installation path: /usr/lib/x86_64-linux-gnu"; CUDNN_INSTALL_PATH=$DEFAULT_CUDNN_PATH;;
        * ) echo "> cuDNN installation path:" $INPUT; CUDNN_INSTALL_PATH=$INPUT;;
      esac
    echo ""
    done

    write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}
    write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "${CUDA_TOOLKIT_PATH}"
    write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "${CUDNN_INSTALL_PATH}"
    write_action_env_to_bazelrc "TF_CUDA_VERSION" "${TF_CUDA_VERSION}"
    write_action_env_to_bazelrc "TF_CUDNN_VERSION" "${TF_CUDNN_VERSION}"

    write_to_bazelrc "test --config=cuda"
    write_to_bazelrc "build --config=cuda"
    write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
    write_to_bazelrc "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"
fi

echo ""
echo "Build configurations successfully written to .bazelrc"
echo ""
