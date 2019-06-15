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

PYTHON_VERSIONS="python2.7 python3.4 python3.5 python3.6"
curl -sSOL https://bootstrap.pypa.io/get-pip.py
add-apt-repository -y ppa:deadsnakes/ppa

for version in ${PYTHON_VERSIONS}; do
    export PYTHON_VERSION=${version}
    apt-get -y -qq update && apt-get -y -qq install ${PYTHON_VERSION}

    ${PYTHON_VERSION} get-pip.py -q
    ${PYTHON_VERSION} -m pip --version

    #Link TF dependency
    yes 'y' | ./configure.sh --quiet

    # Build
    bazel build \
      --noshow_progress \
      --noshow_loading_progress \
      --verbose_failures \
      --test_output=errors \
      build_pip_pkg

    # Package Whl
    bazel-bin/build_pip_pkg artifacts

    # Uncomment and use this command for release branches
    #bazel-bin/build_pip_pkg artifacts
done

# Verify Wheels
./tools/ci_build/builds/wheel_verify.sh