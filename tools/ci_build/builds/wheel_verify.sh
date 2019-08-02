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
ls artifacts/*
for f in artifacts/*.whl; do
  if [[ $(uname) == "Darwin" ]]; then
    python3 -m pip install -q delocate
    delocate-wheel -w wheelhouse  $f
  else
    apt-get -y -qq update && apt-get -y -qq install patchelf
    python3 -m pip install -q auditwheel
    auditwheel -v repair $f
  fi
done
ls wheelhouse/*