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
.PHONY: all

all: code-format sanity-check unit-test

# TODO: install those dependencies in docker image (dockerfile).
install-ci-dependency:
	bash tools/ci_build/install/install_ci_dependency.sh --quiet

code-format: install-ci-dependency
	bash tools/ci_build/code_format.sh --incremental --in-place

sanity-check: install-ci-dependency
	bash tools/ci_build/ci_sanity.sh --incremental

unit-test:
	bash tools/ci_testing/addons_cpu.sh
