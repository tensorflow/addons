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
# Common Bash functions used by build scripts

COLOR_NC='\033[0m'
COLOR_BOLD='\033[1m'
COLOR_LIGHT_GRAY='\033[0;37m'
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'

die() {
    # Print a message and exit with code 1.
    #
    # Usage: die <error_message>
    #   e.g., die "Something bad happened."

    echo $@
    exit 1
}

num_cpus() {
    # Get the number of CPUs
    N_CPUS=$(grep -c ^processor /proc/cpuinfo)
    if [[ -z ${N_CPUS} ]]; then
        die "ERROR: Unable to determine the number of CPUs"
    fi

    echo ${N_CPUS}
}

# List files changed (i.e., added, or revised) from
# the common ancestor of HEAD and the latest master branch.
# Usage: get_changed_files_from_master_branch
get_changed_files_from_master_branch() {
    ANCESTOR=$(git merge-base HEAD master origin/master)
    git diff ${ANCESTOR} --diff-filter=d --name-only "$@"
}

# List bazel files changed that still exist,
# i.e., not removed.
# Usage: get_bazel_files_to_check [--incremental]
get_bazel_files_to_check() {
    if [[ "$1" == "--incremental" ]]; then
        get_changed_files_from_master_branch -- '*BUILD' '*.bazel' '*.bzl'
    elif [[ -z "$1" ]]; then
        find . -name 'BUILD' -o -name '*.bazel' -o -name '*.bzl'
    else
        die "Found unsupported args: $@ for get_bazel_files_to_check."
    fi
}

# List python files changed that still exist,
# i.e., not removed.
# Usage: get_py_files_to_check [--incremental]
get_py_files_to_check() {
    if [[ "$1" == "--incremental" ]]; then
        get_changed_files_from_master_branch -- '*.py'
    elif [[ -z "$1" ]]; then
        find . -name '*.py'
    else
        die "Found unsupported args: $@ for get_py_files_to_check."
    fi
}

# List .h|.cc files changed that still exist,
# i.e., not removed.
# Usage: get_clang_files_to_check [--incremental]
get_clang_files_to_check() {
    if [[ "$1" == "--incremental" ]]; then
        get_changed_files_from_master_branch -- '*.h' '*.cc'
    elif [[ -z "$1" ]]; then
        find . -name '*.h' -o -name '*.cc'
    else
        die "Found unsupported args: $@ for get_clang_files_to_check."
    fi
}

realpath() {
    # Get the real path of a file
    # Usage: realpath <file_path>

    if [[ $# != "1" ]]; then
        die "realpath: incorrect usage"
    fi

    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

to_lower () {
    # Convert string to lower case.
    # Usage: to_lower <string>

    echo "$1" | tr '[:upper:]' '[:lower:]'
}

calc_elapsed_time() {
    # Calculate elapsed time. Takes nanosecond format input of the kind output
    # by date +'%s%N'
    #
    # Usage: calc_elapsed_time <START_TIME> <END_TIME>

    if [[ $# != "2" ]]; then
        die "calc_elapsed_time: incorrect usage"
    fi

    START_TIME=$1
    END_TIME=$2

    if [[ ${START_TIME} == *"N" ]]; then
        # Nanosecond precision not available
        START_TIME=$(echo ${START_TIME} | sed -e 's/N//g')
        END_TIME=$(echo ${END_TIME} | sed -e 's/N//g')
        ELAPSED="$(expr ${END_TIME} - ${START_TIME}) s"
    else
        ELAPSED="$(expr $(expr ${END_TIME} - ${START_TIME}) / 1000000) ms"
    fi

    echo ${ELAPSED}
}
