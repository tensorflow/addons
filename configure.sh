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
# Usage: configure.sh [-q | --quiet] [-p | --pip-index xxx]
#
# Options:
#  -q | --quiet  Give less output.
#  -p | --pip-index using given pip index url to install python package

# getopt part of this code are created with the help of Stack Overflow question
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
# Answer by Robert Siemer:
# https://stackoverflow.com/users/825924/robert-siemer

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=qp:
LONGOPTS=quite,pip-index:

# -use ! and PIPESTATUS to get exit code with errexit set
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

QUIET_FLAG=""
PIP_INDEX_FLAG=""

while true; do
    case "$1" in
        -q|--quite)
            QUIET_FLAG="--quiet"
            shift
            ;;
        -p|--pip-index)
            PIP_INDEX_FLAG="-i $2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Found unsupported args: $@"
            exit 1
            ;;
    esac
done

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

[[ -f .bazelrc ]] && rm .bazelrc
pip install $PIP_INDEX_FLAG $QUIET_FLAG -r requirements.txt

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_LFLAGS:2}
