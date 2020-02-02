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
# Usage: ci_sanity.sh [--incremental] [bazel flags]
#
# Options:
#           run sanity checks: python 2&3 pylint checks and bazel nobuild
#  --incremental  Performs checks incrementally, by using the files changed in
#                 the latest commit

# Current script directory
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
source "${SCRIPT_DIR}/builds/builds_common.sh"

ROOT_DIR=$( cd "$SCRIPT_DIR/../.." && pwd -P )
if [[ ! -d "tensorflow_addons" ]]; then
    echo "ERROR: PWD: $PWD is not project root"
    exit 1
fi


#Check for the bazel cmd status (First arg is error message)
cmd_status(){
    if [[ $? != 0 ]]; then
        echo ""
        echo "FAIL: ${BUILD_CMD}"
        echo "  $1 See lines above for details."
        return 1
    else
        echo ""
        echo "PASS: ${BUILD_CMD}"
        return 0
    fi
}

# Run bazel build --nobuild to test the validity of the BUILD files
do_bazel_nobuild() {
    python3 ./configure.py --quiet

    # Check
    BUILD_TARGET="//tensorflow_addons/..."
    BUILD_CMD="bazel build --nobuild ${BAZEL_FLAGS} -- ${BUILD_TARGET}"
    ${BUILD_CMD}

    cmd_status "This is due to invalid BUILD files."
}

do_check_file_name_test() {
    cd "$ROOT_DIR/tools/ci_build/verify"
    python check_file_name.py
}


# Supply all sanity step commands and descriptions
SANITY_STEPS=("do_bazel_nobuild" "do_check_file_name_test")
SANITY_STEPS_DESC=("bazel nobuild" "Check file names for cases")

INCREMENTAL_FLAG=""
DEFAULT_BAZEL_CONFIGS=""

# Parse command-line arguments
BAZEL_FLAGS=${DEFAULT_BAZEL_CONFIGS}
for arg in "$@"; do
    if [[ "${arg}" == "--incremental" ]]; then
        INCREMENTAL_FLAG="--incremental"
    else
        BAZEL_FLAGS="${BAZEL_FLAGS} ${arg}"
    fi
done


FAIL_COUNTER=0
PASS_COUNTER=0
STEP_EXIT_CODES=()

# Execute all the sanity build steps
COUNTER=0
while [[ ${COUNTER} -lt "${#SANITY_STEPS[@]}" ]]; do
    INDEX=COUNTER
    ((INDEX++))

    echo ""
    echo "=== Sanity check step ${INDEX} of ${#SANITY_STEPS[@]}: "\
         "${SANITY_STEPS[COUNTER]} (${SANITY_STEPS_DESC[COUNTER]}) ==="
    echo ""

    # subshell: don't leak variables or changes of working directory
    (
    ${SANITY_STEPS[COUNTER]} ${INCREMENTAL_FLAG}
    )
    RESULT=$?

    if [[ ${RESULT} != "0" ]]; then
        ((FAIL_COUNTER++))
    else
        ((PASS_COUNTER++))
    fi

    STEP_EXIT_CODES+=(${RESULT})

    echo ""
    ((COUNTER++))
done

# Print summary of build results
COUNTER=0
echo "==== Summary of sanity check results ===="
while [[ ${COUNTER} -lt "${#SANITY_STEPS[@]}" ]]; do
    INDEX=COUNTER
    ((INDEX++))

    echo "${INDEX}. ${SANITY_STEPS[COUNTER]}: ${SANITY_STEPS_DESC[COUNTER]}"
    if [[ ${STEP_EXIT_CODES[COUNTER]} == "0" ]]; then
        printf "  ${COLOR_GREEN}PASS${COLOR_NC}\n"
    else
        printf "  ${COLOR_RED}FAIL${COLOR_NC}\n"
    fi

    ((COUNTER++))
done

echo
echo "${FAIL_COUNTER} failed; ${PASS_COUNTER} passed."

echo
if [[ ${FAIL_COUNTER} == "0" ]]; then
    printf "Sanity checks ${COLOR_GREEN}PASSED${COLOR_NC}\n"
else
    printf "Sanity checks ${COLOR_RED}FAILED${COLOR_NC}\n"
    exit 1
fi
