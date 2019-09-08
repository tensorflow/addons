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
# Usage: ci_sanity.sh [--pep8] [--incremental] [bazel flags]
#
# Options:
#           run sanity checks: python 2&3 pylint checks and bazel nobuild
#  --pep8   run pep8 test only
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

# Run pylint
do_pylint() {
    # Usage: do_pylint (PYTHON2 | PYTHON3) [--incremental]
    #
    # Options:
    #   --incremental  Performs check on only the python files changed in the
    #                  last non-merge git commit.

    # Use this list to whitelist pylint errors
    ERROR_WHITELIST=""

    echo "ERROR_WHITELIST=\"${ERROR_WHITELIST}\""

    if [[ $# != "1" ]] && [[ $# != "2" ]]; then
        echo "Invalid syntax when invoking do_pylint"
        echo "Usage: do_pylint (PYTHON2 | PYTHON3) [--incremental]"
        return 1
    fi

    if [[ $1 == "PYTHON2" ]]; then
        PYLINT_BIN="python2 -m pylint"
    elif [[ $1 == "PYTHON3" ]]; then
        PYLINT_BIN="python3 -m pylint"
    else
        echo "Unrecognized python version (PYTHON2 | PYTHON3): $1"
        return 1
    fi

    PYTHON_SRC_FILES=$(get_py_files_to_check $2)
    if [[ -z ${PYTHON_SRC_FILES} ]]; then
        echo "do_pylint found no Python files to check. Returning."
        return 0
    fi

    PYLINTRC_FILE="${SCRIPT_DIR}/pylintrc"

    if [[ ! -f "${PYLINTRC_FILE}" ]]; then
        die "ERROR: Cannot find pylint rc file at ${PYLINTRC_FILE}"
    fi

    NUM_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)
    NUM_CPUS=$(num_cpus)

    echo "Running pylint on ${NUM_SRC_FILES} files with ${NUM_CPUS} "\
    "parallel jobs..."
    echo ""

    PYLINT_START_TIME=$(date +'%s')
    OUTPUT_FILE="$(mktemp)_pylint_output.log"
    ERRORS_FILE="$(mktemp)_pylint_errors.log"
    NONWL_ERRORS_FILE="$(mktemp)_pylint_nonwl_errors.log"

    rm -rf ${OUTPUT_FILE}
    rm -rf ${ERRORS_FILE}
    rm -rf ${NONWL_ERRORS_FILE}
    touch ${NONWL_ERRORS_FILE}

    ${PYLINT_BIN} --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
        --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} > ${OUTPUT_FILE} 2>&1
    PYLINT_END_TIME=$(date +'%s')

    echo ""
    echo "pylint took $((PYLINT_END_TIME - PYLINT_START_TIME)) s"
    echo ""

    # Report only what we care about
    # Ref https://pylint.readthedocs.io/en/latest/technical_reference/features.html
    # E: all errors
    # W0311 bad-indentation
    # W0312 mixed-indentation
    # C0330 bad-continuation
    # C0301 line-too-long
    # C0326 bad-whitespace
    # W0611 unused-import
    # W0622 redefined-builtin
    grep -E '(\[E|\[W0311|\[W0312|\[C0330|\[C0301|\[C0326|\[W0611|\[W0622)' ${OUTPUT_FILE} > ${ERRORS_FILE}

    N_ERRORS=0
    while read -r LINE; do
        IS_WHITELISTED=0
        for WL_REGEX in ${ERROR_WHITELIST}; do
            if echo ${LINE} | grep -q "${WL_REGEX}"; then
                echo "Found a whitelisted error:"
                echo "  ${LINE}"
                IS_WHITELISTED=1
            fi
        done

        if [[ ${IS_WHITELISTED} == "0" ]]; then
            echo "${LINE}" >> ${NONWL_ERRORS_FILE}
            echo "" >> ${NONWL_ERRORS_FILE}
            ((N_ERRORS++))
        fi
    done <${ERRORS_FILE}

    echo ""
    if [[ ${N_ERRORS} != 0 ]]; then
        echo "FAIL: Found ${N_ERRORS} non-whitelited pylint errors:"
        cat "${NONWL_ERRORS_FILE}"
        return 1
    else
        echo "PASS: No non-whitelisted pylint errors were found."
        return 0
    fi
}

# Run pep8 check
do_pep8() {
    # Usage: do_pep8 [--incremental]
    # Options:
    #   --incremental  Performs check on only the python files changed in the
    #                  last non-merge git commit.

    PEP8_BIN="/usr/local/bin/pep8"
    PEP8_CONFIG_FILE="${SCRIPT_DIR}/pep8"

    if [[ "$1" == "--incremental" ]]; then
        PYTHON_SRC_FILES=$(get_py_files_to_check --incremental)
        NUM_PYTHON_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)

        echo "do_pep8 will perform checks on only the ${NUM_PYTHON_SRC_FILES} "\
             "Python file(s) changed in the last non-merge git commit due to the "\
             "--incremental flag:"
        echo "${PYTHON_SRC_FILES}"
        echo ""
    else
        PYTHON_SRC_FILES=$(get_py_files_to_check)
    fi

    if [[ -z ${PYTHON_SRC_FILES} ]]; then
        echo "do_pep8 found no Python files to check. Returning."
        return 0
    fi

    if [[ ! -f "${PEP8_CONFIG_FILE}" ]]; then
        die "ERROR: Cannot find pep8 config file at ${PEP8_CONFIG_FILE}"
    fi
    echo "See \"${PEP8_CONFIG_FILE}\" for pep8 config( e.g., ignored errors)"

    NUM_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)

    echo "Running pep8 on ${NUM_SRC_FILES} files"
    echo ""

    PEP8_START_TIME=$(date +'%s')
    PEP8_OUTPUT_FILE="$(mktemp)_pep8_output.log"

    rm -rf ${PEP8_OUTPUT_FILE}

    ${PEP8_BIN} --config="${PEP8_CONFIG_FILE}" --statistics \
        ${PYTHON_SRC_FILES} 2>&1 | tee ${PEP8_OUTPUT_FILE}
    PEP8_END_TIME=$(date +'%s')

    echo ""
    echo "pep8 took $((PEP8_END_TIME - PEP8_START_TIME)) s"
    echo ""

    if [[ -s ${PEP8_OUTPUT_FILE} ]]; then
        echo "FAIL: pep8 found above errors and/or warnings."
        return 1
    else
        echo "PASS: No pep8 errors or warnings were found"
        return 0
    fi
}


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
    # Use default configuration here.
    yes 'y' | ./configure.sh --quiet

    # Check
    BUILD_TARGET="//tensorflow_addons/..."
    BUILD_CMD="bazel build --nobuild ${BAZEL_FLAGS} -- ${BUILD_TARGET}"
    ${BUILD_CMD}

    cmd_status "This is due to invalid BUILD files."
}

do_check_futures_test() {
    cd "$ROOT_DIR/tools/test"
    python check_futures_test.py
}

do_check_file_name_test() {
    cd "$ROOT_DIR/tools/test"
    python file_name_test.py
}

do_check_code_format_test() {
    CHECK_CMD="$SCRIPT_DIR/code_format.sh $1"
    ${CHECK_CMD}
}

# Supply all sanity step commands and descriptions
SANITY_STEPS=("do_check_code_format_test" "do_pylint PYTHON2" "do_pylint PYTHON3" "do_check_futures_test" "do_bazel_nobuild" "do_check_file_name_test")
SANITY_STEPS_DESC=("Check code style" "Python 2 pylint" "Python 3 pylint" "Check that python files have certain __future__ imports" "bazel nobuild" "Check file names for cases")

INCREMENTAL_FLAG=""
DEFAULT_BAZEL_CONFIGS=""

# Parse command-line arguments
BAZEL_FLAGS=${DEFAULT_BAZEL_CONFIGS}
for arg in "$@"; do
    if [[ "${arg}" == "--pep8" ]]; then
        # Only run pep8 test if "--pep8" option supplied
        SANITY_STEPS=("do_pep8")
        SANITY_STEPS_DESC=("pep8 test")
    elif [[ "${arg}" == "--incremental" ]]; then
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
