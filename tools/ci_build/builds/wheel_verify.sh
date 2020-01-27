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

set -e -x

if [[ $(uname) == "Darwin" ]]; then
    CMD="delocate-wheel -w wheelhouse"
elif [[ $(uname) == "Linux" ]]; then
    apt-get -y -qq update && apt-get install patchelf
    python3 -m pip install -U auditwheel==2.0.0
    tools/ci_build/builds/tf_auditwheel_patch.sh
    CMD="auditwheel repair --plat manylinux2010_x86_64"
fi

ls artifacts/*
for f in artifacts/*.whl; do
    $CMD $f
done
ls wheelhouse/*