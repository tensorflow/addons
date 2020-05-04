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

# usage: bash tools/update_release_version.sh <release_number>

sed -ri "s/(TF_VERSION=|tensorflow(-cpu)*(~|=)=|tf-version: \[')[0-9]+[a-zA-Z0-9_.-]+/\1$1/g" \
	.github/workflows/release.yml \
	CONTRIBUTING.md \
	tools/docker/cpu_tests.Dockerfile \
	tools/install_deps/tensorflow-cpu.txt \
	tools/install_deps/tensorflow.txt \
	tools/run_gpu_tests.sh
