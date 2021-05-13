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

# Usage
if [ $# -lt 1 ]; then
	echo "Usage: bash tools/update_release_version.sh <list_of_release_numbers>"
	echo "e.g. bash tools/update_release_version.sh 2.3.0 2.3.1"
	exit 1
fi

last_version=${BASH_ARGV[0]}
tf_version=''
for ver in $@
do
    if [ -z $tf_version ]; then
		tf_version="'$ver'"
	else 
	    tf_version="$tf_version, '$ver'"    
	fi
done
echo $tf_version
echo $last_version
sed -ri "s/(tf-version: \[)'.+'/\1$tf_version/g" \
	.github/workflows/release.yml
sed -ri "s/(tensorflow(-cpu)*(~|=)=)[0-9]+[a-zA-Z0-9_.-]+/\1$1/g" \
	CONTRIBUTING.md \
	tools/install_deps/tensorflow-cpu.txt \
	tools/install_deps/tensorflow.txt
sed -ri "s/(TF_VERSION=)\S+/\1$last_version/g" \
	tools/docker/cpu_tests.Dockerfile \
	tools/run_gpu_tests.sh \
	tools/build_dev_container.sh
