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

PYTHON_VERSIONS="2.7.15 3.4.9 3.5.6 3.6.6"
curl -sSOL https://bootstrap.pypa.io/get-pip.py

wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-darwin-x86_64.sh
chmod +x bazel-0.24.1-installer-darwin-x86_64.sh
./bazel-0.24.1-installer-darwin-x86_64.sh --user

export PATH="$PATH:$HOME/bin"

eval "$(pyenv init -)"

for version in ${PYTHON_VERSIONS}; do
    export PYENV_VERSION=${version}
    pyenv install -s $PYENV_VERSION

    python get-pip.py -q
    python -m pip install -q delocate

    #Link TF dependency
    yes 'y' | sudo ./configure.sh

    # Build
    bazel build \
      --noshow_progress \
      --noshow_loading_progress \
      --verbose_failures \
      --test_output=errors \
      build_pip_pkg

    # Package Whl
    bazel-bin/build_pip_pkg artifacts --nightly

    # Uncomment and use this command for release branches
    #bazel-bin/build_pip_pkg artifacts
done

## Verify Wheel
./tools/ci_build/builds/wheel_verify.sh
