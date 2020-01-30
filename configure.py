# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# Usage: configure.py [--quiet] [--no-deps]
#
# Options:
#  --quiet  Give less output.
#  --no-deps  Don't install Python dependencies


import argparse
import os
import platform
import subprocess
import sys
import logging

_DEFAULT_CUDA_VERISON = "10.1"
_DEFAULT_CUDA_PATH = "/usr/local/cuda"
_DEFAULT_CUDNN_VERSION = "7"
_DEFAULT_CUDNN_PATH = "/usr/lib/x86_64-linux-gnu"

_TFA_BAZELRC = ".bazelrc"
_TF_NEED_CUDA = os.getenv("TF_NEED_CUDA")
_TF_CUDA_VERSION = os.getenv("TF_CUDA_VERSION")
_CUDA_TOOLKIT_PATH = os.getenv("CUDA_TOOLKIT_PATH")
_TF_CUDNN_VERSION = os.getenv("TF_CUDNN_VERSION")
_CUDNN_INSTALL_PATH = os.getenv("CUDNN_INSTALL_PATH")


# Writes variables to bazelrc file
def write_to_bazelrc(line):
    with open(_TFA_BAZELRC, "a") as f:
        f.write(line + "\n")


def write_action_env_to_bazelrc(var_name, var):
    write_to_bazelrc('build --action_env %s="%s"' % (var_name, str(var)))


def is_linux():
    return platform.system() == "Linux"


def is_macos():
    return platform.system() == "Darwin"


def is_windows():
    return platform.system() == "Windows"


def is_ppc64le():
    return platform.machine() == "ppc64le"


def get_input(question):
    try:
        try:
            answer = raw_input(question)
        except NameError:
            answer = input(question)  # pylint: disable=bad-builtin
    except EOFError:
        answer = ""
    return answer


# Converts the linkflag namespec to the full shared library name
def generate_shared_lib_name(namespec):
    if is_macos():
        # MacOS
        return "lib" + namespec[1][2:] + ".dylib"
    elif is_windows():
        # Windows
        return "_pywrap_tensorflow_internal.lib"
    else:
        # Linux
        return namespec[1][3:]


print()
print("Configuring TensorFlow Addons to be built from source...")

_PIP_INSTALL_OPTS = ["--upgrade"]
parser = argparse.ArgumentParser()
parser.add_argument("--quiet", action="store_true", help="Give less output.")
parser.add_argument(
    "--no-deps",
    action="store_true",
    help="Do not check and install Python dependencies.",
)
args = parser.parse_args()
if args.quiet:
    _PIP_INSTALL_OPTS.append("--quiet")

_PYTHON_PATH = sys.executable
with open("requirements.txt") as f:
    _REQUIRED_PKG = f.read().splitlines()

with open("build_deps/build-requirements.txt") as f:
    _REQUIRED_PKG.extend(f.read().splitlines())

print()
if args.no_deps:
    print("> Using pre-installed Tensorflow.")
else:
    print("> Installing", _REQUIRED_PKG)
    install_cmd = [_PYTHON_PATH, "-m", "pip", "install"]
    install_cmd.extend(_PIP_INSTALL_OPTS)
    install_cmd.extend(_REQUIRED_PKG)
    subprocess.check_call(install_cmd)

if os.path.isfile(_TFA_BAZELRC):
    os.remove(_TFA_BAZELRC)

logging.disable(logging.WARNING)

import tensorflow as tf  # noqa: E402 module level import not at top of file

_TF_CFLAGS = tf.sysconfig.get_compile_flags()
_TF_LFLAGS = tf.sysconfig.get_link_flags()
_TF_CXX11_ABI_FLAG = tf.sysconfig.CXX11_ABI_FLAG

_TF_SHARED_LIBRARY_NAME = generate_shared_lib_name(_TF_LFLAGS)
_TF_HEADER_DIR = _TF_CFLAGS[0][2:]

# OS Specific parsing
if is_windows():
    _TF_SHARED_LIBRARY_DIR = _TF_CFLAGS[0][2:-7] + "python"
    _TF_SHARED_LIBRARY_DIR = _TF_SHARED_LIBRARY_DIR.replace("\\", "/")

    _TF_HEADER_DIR = _TF_HEADER_DIR.replace("\\", "/")
else:
    _TF_SHARED_LIBRARY_DIR = _TF_LFLAGS[0][2:]

write_action_env_to_bazelrc("TF_HEADER_DIR", _TF_HEADER_DIR)
write_action_env_to_bazelrc("TF_SHARED_LIBRARY_DIR", _TF_SHARED_LIBRARY_DIR)
write_action_env_to_bazelrc("TF_SHARED_LIBRARY_NAME", _TF_SHARED_LIBRARY_NAME)
write_action_env_to_bazelrc("TF_CXX11_ABI_FLAG", _TF_CXX11_ABI_FLAG)

write_to_bazelrc("build --spawn_strategy=standalone")
write_to_bazelrc("build --strategy=Genrule=standalone")
write_to_bazelrc("build -c opt")


while _TF_NEED_CUDA is None:
    print()
    INPUT = get_input("Do you want to build GPU ops? [y/N] ")
    if INPUT in ("Y", "y"):
        print("> Building GPU & CPU ops")
        _TF_NEED_CUDA = "1"
    elif INPUT in ("N", "n", ""):
        print("> Building only CPU ops")
        _TF_NEED_CUDA = "0"
    else:
        print("Invalid selection: {}".format(INPUT))

if _TF_NEED_CUDA == "1":
    print()
    print("Configuring GPU setup...")

    if _TF_CUDA_VERSION is None:
        INPUT = get_input(
            "Please specify the CUDA version [Default is {}]: ".format(
                _DEFAULT_CUDA_VERISON
            )
        )
        _TF_CUDA_VERSION = INPUT if INPUT else _DEFAULT_CUDA_VERISON
    print("> Using CUDA version: {}".format(_TF_CUDA_VERSION))
    print()

    if _CUDA_TOOLKIT_PATH is None:
        INPUT = get_input(
            "Please specify the location of CUDA. [Default is {}]: ".format(
                _DEFAULT_CUDA_PATH
            )
        )
        _CUDA_TOOLKIT_PATH = INPUT if INPUT else _DEFAULT_CUDA_PATH
    print("> CUDA installation path: {}".format(_CUDA_TOOLKIT_PATH))
    print()

    if _TF_CUDNN_VERSION is None:
        INPUT = get_input(
            "Please specify the cuDNN major version [Default is {}]: ".format(
                _DEFAULT_CUDNN_VERSION
            )
        )
        _TF_CUDNN_VERSION = INPUT if INPUT else _DEFAULT_CUDNN_VERSION
    print("> Using cuDNN version: {}".format(_TF_CUDNN_VERSION))
    print()

    if _CUDNN_INSTALL_PATH is None:
        INPUT = get_input(
            "Please specify the location of cuDNN installation. [Default is {}]: ".format(
                _DEFAULT_CUDNN_PATH
            )
        )
        _CUDNN_INSTALL_PATH = INPUT if INPUT else _DEFAULT_CUDNN_PATH
    print("> cuDNN installation path: {}".format(_CUDNN_INSTALL_PATH))
    print()

    write_action_env_to_bazelrc("TF_NEED_CUDA", _TF_NEED_CUDA)
    write_action_env_to_bazelrc("CUDA_TOOLKIT_PATH", _CUDA_TOOLKIT_PATH)
    write_action_env_to_bazelrc("CUDNN_INSTALL_PATH", _CUDNN_INSTALL_PATH)
    write_action_env_to_bazelrc("TF_CUDA_VERSION", _TF_CUDA_VERSION)
    write_action_env_to_bazelrc("TF_CUDNN_VERSION", _TF_CUDNN_VERSION)

    write_to_bazelrc("test --config=cuda")
    write_to_bazelrc("build --config=cuda")
    write_to_bazelrc(
        "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
    )
    write_to_bazelrc(
        "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"
    )

print()
print("Build configurations successfully written to {}".format(_TFA_BAZELRC))
print()


def main():
    pass


if __name__ == "__main__":
    main()
