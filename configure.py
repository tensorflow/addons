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
# Usage: python configure.py
#


import os
import pathlib
import platform
import logging

import tensorflow as tf

_DEFAULT_CUDA_VERISON = "10.1"
_DEFAULT_CUDNN_VERSION = "7"
_TFA_BAZELRC = ".bazelrc"


# Writes variables to bazelrc file
def write_to_bazelrc(line):
    with open(_TFA_BAZELRC, "a") as f:
        f.write(line + "\n")


def write_action_env_to_bazelrc(var_name, var):
    write_to_bazelrc('build --action_env %s="%s"' % (var_name, str(var)))


def is_macos():
    return platform.system() == "Darwin"


def is_windows():
    return platform.system() == "Windows"


def get_input(question):
    try:
        return input(question)
    except EOFError:
        return ""


def get_tf_header_dir():
    import tensorflow as tf

    tf_header_dir = tf.sysconfig.get_compile_flags()[0][2:]
    if is_windows():
        tf_header_dir = tf_header_dir.replace("\\", "/")
    return tf_header_dir


def get_tf_shared_lib_dir():
    import tensorflow as tf

    # OS Specific parsing
    if is_windows():
        tf_shared_lib_dir = tf.sysconfig.get_compile_flags()[0][2:-7] + "python"
        return tf_shared_lib_dir.replace("\\", "/")
    else:
        return tf.sysconfig.get_link_flags()[0][2:]


# Converts the linkflag namespec to the full shared library name
def get_shared_lib_name():
    import tensorflow as tf

    namespec = tf.sysconfig.get_link_flags()
    if is_macos():
        # MacOS
        return "lib" + namespec[1][2:] + ".dylib"
    elif is_windows():
        # Windows
        return "_pywrap_tensorflow_internal.lib"
    else:
        # Linux
        return namespec[1][3:]


def create_build_configuration():
    print()
    print("Configuring TensorFlow Addons to be built from source...")

    if os.path.isfile(_TFA_BAZELRC):
        os.remove(_TFA_BAZELRC)

    logging.disable(logging.WARNING)

    write_action_env_to_bazelrc("TF_HEADER_DIR", get_tf_header_dir())
    write_action_env_to_bazelrc("TF_SHARED_LIBRARY_DIR", get_tf_shared_lib_dir())
    write_action_env_to_bazelrc("TF_SHARED_LIBRARY_NAME", get_shared_lib_name())
    write_action_env_to_bazelrc("TF_CXX11_ABI_FLAG", tf.sysconfig.CXX11_ABI_FLAG)

    write_to_bazelrc("build --spawn_strategy=standalone")
    write_to_bazelrc("build --strategy=Genrule=standalone")
    write_to_bazelrc("build -c opt")

    _TF_NEED_CUDA = os.getenv("TF_NEED_CUDA")

    while _TF_NEED_CUDA is None:
        print()
        answer = get_input("Do you want to build GPU ops? [y/N] ")
        if answer in ("Y", "y"):
            print("> Building GPU & CPU ops")
            _TF_NEED_CUDA = "1"
        elif answer in ("N", "n", ""):
            print("> Building only CPU ops")
            _TF_NEED_CUDA = "0"
        else:
            print("Invalid selection:", answer)

    if _TF_NEED_CUDA == "1":
        configure_cuda()

    print()
    print("Build configurations successfully written to", _TFA_BAZELRC)
    print(pathlib.Path(_TFA_BAZELRC).read_text())
    print()


def get_cuda_toolkit_path():
    default = "/usr/local/cuda"
    cuda_toolkit_path = os.getenv("CUDA_TOOLKIT_PATH")
    if cuda_toolkit_path is None:
        answer = get_input(
            "Please specify the location of CUDA. [Default is {}]: ".format(default)
        )
        cuda_toolkit_path = answer or default
    print("> CUDA installation path:", cuda_toolkit_path)
    print()
    return cuda_toolkit_path


def get_cudnn_install_path():
    default = "/usr/lib/x86_64-linux-gnu"
    cudnn_install_path = os.getenv("CUDNN_INSTALL_PATH")
    if cudnn_install_path is None:
        answer = get_input(
            "Please specify the location of cuDNN installation. [Default is {}]: ".format(
                default
            )
        )
        cudnn_install_path = answer or default
    print("> cuDNN installation path:", cudnn_install_path)
    print()
    return cudnn_install_path


def configure_cuda():
    _TF_CUDA_VERSION = os.getenv("TF_CUDA_VERSION")
    _TF_CUDNN_VERSION = os.getenv("TF_CUDNN_VERSION")

    print()
    print("Configuring GPU setup...")

    if _TF_CUDA_VERSION is None:
        answer = get_input(
            "Please specify the CUDA version [Default is {}]: ".format(
                _DEFAULT_CUDA_VERISON
            )
        )
        _TF_CUDA_VERSION = answer or _DEFAULT_CUDA_VERISON
    print("> Using CUDA version:", _TF_CUDA_VERSION)
    print()

    if _TF_CUDNN_VERSION is None:
        answer = get_input(
            "Please specify the cuDNN major version [Default is {}]: ".format(
                _DEFAULT_CUDNN_VERSION
            )
        )
        _TF_CUDNN_VERSION = answer or _DEFAULT_CUDNN_VERSION
    print("> Using cuDNN version:", _TF_CUDNN_VERSION)
    print()

    write_action_env_to_bazelrc("TF_NEED_CUDA", "1")
    write_action_env_to_bazelrc("CUDA_TOOLKIT_PATH", get_cuda_toolkit_path())
    write_action_env_to_bazelrc("CUDNN_INSTALL_PATH", get_cudnn_install_path())
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


if __name__ == "__main__":
    create_build_configuration()
