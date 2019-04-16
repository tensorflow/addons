#!/bin/bash
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
    [yY]*) echo "Install...";;
    * ) echo "Goodbye!"; exit;;
esac
pip install $QUIET_FLAG -r requirements.txt

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_LFLAGS:2}
