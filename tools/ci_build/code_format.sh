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
# Usage: code_format.sh [--incremental] [--in-place]
#
# Options:
#  --incremental  Performs checks incrementally, by using the files changed in
#                 the latest commit
#  --in-place  make changes to files in place

# Current script directory
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
source "${SCRIPT_DIR}/builds/builds_common.sh"

ROOT_DIR=$( cd "$SCRIPT_DIR/../.." && pwd -P )
if [[ ! -d "tensorflow_addons" ]]; then
    echo "ERROR: PWD: $PWD is not project root"
    exit 1
fi

# Parse command-line arguments
INCREMENTAL_FLAG=""
IN_PLACE_FLAG=""
UNRESOLVED_ARGS=""

for arg in "$@"; do
    if [[ "${arg}" == "--incremental" ]]; then
        INCREMENTAL_FLAG="--incremental"
    elif [[ "${arg}" == "--in-place" ]]; then
        IN_PLACE_FLAG="--in-place"
    else
        UNRESOLVED_ARGS="${UNRESOLVED_ARGS} ${arg}"
    fi
done

if [[ ! -z "$UNRESOLVED_ARGS" ]]; then
    die "ERROR: Found unsupported args: $UNRESOLVED_ARGS"
fi

do_bazel_config_format_check() {
    BUILD_FILES=$(get_bazel_files_to_check $INCREMENTAL_FLAG)
    if [[ -z $BUILD_FILES ]]; then
        echo "do_bazel_config_format_check will NOT run due to"\
             "the absence of code changes."
        return 0
    fi

    NUM_BUILD_FILES=$(echo ${BUILD_FILES} | wc -w)
    echo "Running do_buildifier on ${NUM_BUILD_FILES} files"
    echo ""

    if [[ ! -z $IN_PLACE_FLAG ]]; then
        echo "Auto format..."
        buildifier -v -mode=fix ${BUILD_FILES}
    fi

    BUILDIFIER_START_TIME=$(date +'%s')
    BUILDIFIER_OUTPUT_FILE="$(mktemp)_buildifier_output.log"

    rm -rf ${BUILDIFIER_OUTPUT_FILE}

    buildifier -showlog -v -mode=check \
        ${BUILD_FILES} 2>&1 | tee ${BUILDIFIER_OUTPUT_FILE}
    BUILDIFIER_END_TIME=$(date +'%s')

    echo ""
    echo "buildifier took $((BUILDIFIER_END_TIME - BUILDIFIER_START_TIME)) s"
    echo ""

    if [[ -s ${BUILDIFIER_OUTPUT_FILE} ]]; then
        echo "FAIL: buildifier found errors and/or warnings in above BUILD files."
        echo "buildifier suggested the following changes:"
        buildifier -showlog -v -mode=diff ${BUILD_FILES}
        echo "Please fix manually or run buildifier <file> to auto-fix."
        echo "Bazel configuration format check fails."
        return 1
    else
        echo "Bazel configuration format check success."
        return 0
    fi
}

do_python_format_check() {
    PYTHON_SRC_FILES=$(get_py_files_to_check $INCREMENTAL_FLAG)
    if [[ -z $PYTHON_SRC_FILES ]]; then
        echo "do_python_format_check will NOT run due to"\
             "the absence of code changes."
        return 0
    fi

    YAPFRC_FILE="${SCRIPT_DIR}/yapfrc"
    if [[ ! -f "${YAPFRC_FILE}" ]]; then
        die "ERROR: Cannot find yapf rc file at ${YAPFRC_FILE}"
    fi
    YAPF_OPTS="--style=$YAPFRC_FILE --parallel"

    echo $PYTHON_SRC_FILES
    if [[ ! -z $IN_PLACE_FLAG ]]; then
        echo "Auto format..."
        yapf $YAPF_OPTS --in-place --verbose $PYTHON_SRC_FILES
        docformatter --in-place $PYTHON_SRC_FILES
    fi

    UNFORMATTED_CODES=$(yapf $YAPF_OPTS --diff $PYTHON_SRC_FILES)
    if [[ $? != "0" || ! -z "$UNFORMATTED_CODES" ]]; then
        echo "Find unformatted codes:"
        echo "$UNFORMATTED_CODES"
        echo "Python format check fails."
        return 1
    else
        echo "Python format check success."
        return 0
    fi
}

do_clang_format_check() {
    CLANG_SRC_FILES=$(get_clang_files_to_check $INCREMENTAL_FLAG)
    if [[ -z $CLANG_SRC_FILES ]]; then
        echo "do_clang_format_check will NOT run due to"\
             "the absence of code changes."
        return 0
    fi

    CLANG_FORMAT=${CLANG_FORMAT:-clang-format-3.8}
    CLANG_FORMAT_OPTS="--style=google"

    if [[ ! -z $IN_PLACE_FLAG ]]; then
        echo "Auto format..."
        $CLANG_FORMAT $CLANG_FORMAT_OPTS -i $CLANG_SRC_FILES
    fi

    success=1
    for filename in $CLANG_SRC_FILES; do
        $CLANG_FORMAT $CLANG_FORMAT_OPTS $filename | diff $filename - > /dev/null
        if [ ! $? -eq 0 ]; then
            success=0
            echo "File $filename is not properly formatted with clang-format --style=google"
        fi
    done

    if [ $success == 0 ]; then
        echo "Clang format check fails."
        return 1
    else
        echo "Clang format check success."
        return 0
    fi
}

# Supply all auto format step commands and descriptions
FORMAT_STEPS=("do_bazel_config_format_check" "do_python_format_check" "do_clang_format_check")
FORMAT_STEPS_DESC=("Check Bazel file format" "Check python file format" "Check  C++ file format")

FAIL_COUNTER=0
PASS_COUNTER=0
STEP_EXIT_CODES=()

# Execute all the auto format steps
COUNTER=0
while [[ ${COUNTER} -lt "${#FORMAT_STEPS[@]}" ]]; do
    INDEX=COUNTER
    ((INDEX++))

    echo ""
    echo "--- Format check step ${INDEX} of ${#FORMAT_STEPS[@]}: "\
         "${FORMAT_STEPS[COUNTER]} (${FORMAT_STEPS_DESC[COUNTER]}) ---"
    echo ""

    # subshell: don't leak variables or changes of working directory
    (
    ${FORMAT_STEPS[COUNTER]}
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

# Print summary of results
COUNTER=0
echo "---- Summary of format check results ----"
while [[ ${COUNTER} -lt "${#FORMAT_STEPS[@]}" ]]; do
    INDEX=COUNTER
    ((INDEX++))

    echo "${INDEX}. ${FORMAT_STEPS[COUNTER]}: ${FORMAT_STEPS_DESC[COUNTER]}"
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
    printf "Format checks ${COLOR_GREEN}PASSED${COLOR_NC}\n"
else
    printf "Use ${COLOR_GREEN}make format${COLOR_NC} command to format codes automatically\n"
    printf "Format checks ${COLOR_RED}FAILED${COLOR_NC}\n"
    exit 1
fi
