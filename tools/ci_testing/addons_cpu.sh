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
set -x

if [[ $1 == "--py3" ]]; then
    echo "Using python3..."
    docker run --rm -t -v ${PWD}:/addons -w /addons tensorflow/tensorflow:custom-op /bin/bash -c "ln -sf /usr/bin/python3 /usr/bin/python && make unit-test"

elif [[ $1 == "--py2" ]]; then
    echo "Using python2..."
    docker run --rm -t -v ${PWD}:/addons -w /addons tensorflow/tensorflow:custom-op /bin/bash -c "make unit-test"

elif [[ -z "$1" ]]; then
    echo "Using python2..."
    docker run --rm -t -v ${PWD}:/addons -w /addons tensorflow/tensorflow:custom-op /bin/bash -c "make unit-test"

else
    echo "Found unsupported args: $@"
    exit 1
fi
