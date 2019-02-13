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
function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Parse arguments
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        --py_version)
            PY_VERSION=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done


if [ "$PY_VERSION" -eq "3" ]; then
    echo ${PY_VERSION}
    PY_PATH=`which python3`
    PIP_PATH=`which pip3`
else

    PY_PATH=`which python`
    PIP_PATH=`which pip`
fi

rm .bazelrc
if ${PY_PATH} -c "import tensorflow" &> /dev/null; then
    echo 'using installed tensorflow'
else
    ${PIP_PATH} install tf-nightly-2.0-preview
fi

TF_CFLAGS=( $(${PY_PATH} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(${PY_PATH} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_LFLAGS:2}