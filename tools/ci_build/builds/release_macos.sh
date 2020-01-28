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
set -e -x

# No GPU support for MacOS
export TF_NEED_CUDA="0"

# Install delocate
python3 -m pip install -q delocate

#Link TF dependency
yes 'y' | ./configure.sh --quiet

# Build
bazel build \
  -c opt \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

# Package Whl
bazel-bin/build_pip_pkg artifacts --nightly

# Uncomment and use this command for release branches
#bazel-bin/build_pip_pkg artifacts


## Verify Wheel
./tools/ci_build/builds/wheel_verify.sh