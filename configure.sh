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

QUIET_FLAG=""
if [[ $1 == "--quiet" ]]; then
    QUIET_FLAG="--quiet"
elif [[ ! -z "$1" ]]; then
    echo "Found unsupported args: $@"
    exit 1
fi

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

[[ -f .bazelrc ]] && rm .bazelrc
read -r -p "Tensorflow will be upgraded to 2.0. Are You Sure? [Y/n] " reply
case $reply in
    [yY]*) echo "Installing...";;
    * ) echo "Goodbye!"; exit;;
esac
${PYTHON_VERSION:=python} -m pip install $QUIET_FLAG -r requirements.txt

TF_CFLAGS=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"
TF_CXX11_ABI_FLAG=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)') )

SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_CXX11_ABI_FLAG}

write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
write_to_bazelrc "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}

# TODO(yifeif): do not hardcode path
if [[ "$TF_NEED_CUDA" == "1" ]]; then
    # TODO: use CUDA_HOME here?
    write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "/usr/lib/x86_64-linux-gnu"
    write_action_env_to_bazelrc "TF_CUDA_VERSION" "10.0"
    write_action_env_to_bazelrc "TF_CUDNN_VERSION" "7"
    write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "${CUDA_HOME:=/usr/local/cuda}"
    write_to_bazelrc "build --config=cuda"
    write_to_bazelrc "test --config=cuda"
fi
