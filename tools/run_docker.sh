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
#
# ==============================================================================
# Detect the project root path.
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
ROOT_DIR=$( cd "$SCRIPT_DIR/.." && pwd -P )
if [[ ! -d "${ROOT_DIR}/tensorflow_addons" ]]; then
    echo "ERROR: $ROOT_DIR is not project root"
    exit 1
fi

SYSTEM=linux
DEVICE=cpu
PYTHON=py3

while getopts ":s:d:p:c:h" opt; do
    case ${opt} in
        s) SYSTEM=$OPTARG;;
        d) DEVICE=$OPTARG;;
        p) PYTHON=$OPTARG;;
        c) COMMAND=$OPTARG;;
        h)
            echo -n "usage: run_docker.sh [-s SYSTEM] [-d DEVICE] "
            echo "[-p PYTHON] -c string"
            echo "available commands:"
            echo "    -s    select operating system: 'linux'"
            echo "    -d    select device: 'cpu', 'gpu'"
            echo "    -p    select python version: 'py2', 'py3'"
            echo "    -c    command string, eg: 'make unit-test'"
            echo "    -h    print this help and exit"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Invalid option: Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

if [[ "$SYSTEM" != "linux" ]]; then
    echo "System $SYSTEM is not supported yet"
    exit 1
fi

DOCKER_OPTS=''
case ${DEVICE} in
    cpu)
        DOCKER_IMAGE=tensorflow/tensorflow:custom-op
        ;;
    gpu)
        DOCKER_IMAGE=tensorflow/tensorflow:custom-op-gpu
        DOCKER_OPTS="--runtime=nvidia ${DOCKER_OPTS}"
        ;;
    *)
        echo "Invalid or missing device $OPTARG"
        exit 1
        ;;
esac

case ${PYTHON} in
    py2) PYTHON_LIB=/usr/bin/python2;;
    py3) PYTHON_LIB=/usr/bin/python3;;
    *)
        echo "Invalid or missing python $OPTARG"
        exit 1
        ;;
esac

if [[ -z "${COMMAND}" ]]; then
    echo "command string cannot be empty"
    exit 1
fi

DOCKER_CMD="ln -sf ${PYTHON_LIB} /usr/bin/python && ${COMMAND}"
echo "Docker image: ${DOCKER_IMAGE}"
echo "Docker command: ${DOCKER_CMD}"
docker run ${DOCKER_OPTS}                   \
    --network=host                          \
    --rm -v ${ROOT_DIR}:/addons -w /addons  \
    ${DOCKER_IMAGE}                         \
    /bin/bash -c "${DOCKER_CMD}"
