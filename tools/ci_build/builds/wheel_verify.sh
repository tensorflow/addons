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

set -e

if [[ $(uname) == "Darwin" ]]; then
    CMD="delocate-wheel -w wheelhouse"
else
    pip3 install auditwheel==1.5.0
    sudo pip3 install wheel==0.31.1
    CMD="auditwheel repair"

#    pip3 install auditwheel==2.0.0
#    LD_PATH="$(cat .bazelrc | grep TF_SHARED_LIBRARY_DIR | sed 's/"//g' | awk -F'=' '{print $2}')"
#    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_PATH
#    CMD="auditwheel repair --plat manylinux2010_x86_64"
fi

ls artifacts/*
for f in artifacts/*.whl; do
    $CMD $f
done
ls wheelhouse/*