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
set -e
set -x

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

if is_windows; then
  PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.exe.runfiles/__main__/"
else
  PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/__main__/"
fi

function abspath() {
  cd "$(dirname $1)"
  echo "$PWD/$(basename $1)"
  cd "$OLDPWD"
}

function main() {
  DEST=${1}
  NIGHTLY_FLAG=${2}

  if [[ -z ${DEST} ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  # Check if python3 is available. On Windows VM it is not.
  if [ -x "$(command -v python3)" ]; then
      _PYTHON_BINARY=python3
    else
      _PYTHON_BINARY=python
  fi

  mkdir -p ${DEST}
  DEST=$(abspath "${DEST}")
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  echo "=== Copy TensorFlow Addons files"

  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}LICENSE "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}requirements.txt "${TMPDIR}"
  touch ${TMPDIR}/stub.cc

  if is_windows; then
    from=$(cygpath -w ${PIP_FILE_PREFIX}tensorflow_addons)
    to=$(cygpath -w "${TMPDIR}"/tensorflow_addons)
    start robocopy //S "${from}" "${to}" //xf *_test.py
    sleep 5
  else
    rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX}tensorflow_addons "${TMPDIR}"
  fi

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"


  BUILD_CMD="setup.py bdist_wheel"
  if is_macos; then
    BUILD_CMD="${BUILD_CMD} --plat-name macosx_10_13_x86_64"
  fi

  if [[ -z ${NIGHTLY_FLAG} ]]; then
    # Windows has issues with locking library files for deletion so do not fail here
    ${_PYTHON_BINARY} ${BUILD_CMD} || true
  else
    ${_PYTHON_BINARY} ${BUILD_CMD} ${NIGHTLY_FLAG} || true
  fi

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"