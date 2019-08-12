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
#
# Usage: install_ci_dependency.sh [--quiet]
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

# Current script directory
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )

ROOT_DIR=$( cd "$SCRIPT_DIR/../../.." && pwd -P )
if [[ ! -d "tensorflow_addons" ]]; then
    echo "ERROR: PWD: $PWD is not project root"
    exit 1
fi

# Download buildifier.
wget https://github.com/bazelbuild/buildtools/releases/download/0.4.5/buildifier
chmod +x buildifier
sudo mv buildifier /usr/local/bin/.

# Install python dependencies:
CI_REQUIREMENT="$SCRIPT_DIR/ci_requirements.txt"
pip install ${QUIET_FLAG} -r ${CI_REQUIREMENT}

# Check clang-format:
CLANG_FORMAT=${CLANG_FORMAT:-clang-format-3.8}
which ${CLANG_FORMAT} > /dev/null
if [[ $? != "0" ]]; then
    echo "ERROR: cannot find clang-format, please install it."
    exit 1
fi
