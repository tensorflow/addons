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


# Ensure TensorFlow is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import tensorflow, too.
def _ensure_tf_install():
    """Attempt to import tensorflow, and ensure its version is sufficient.
    Raises:
      ImportError: if either tensorflow is not importable or its version is
      inadequate.
    """
    import tensorflow as tf
    import distutils.version

    #
    # Update this whenever we need to depend on a newer TensorFlow release.
    #
    required_tensorflow_version = "2.1.0"

    if distutils.version.LooseVersion(tf.__version__) < distutils.version.LooseVersion(
        required_tensorflow_version
    ):
        raise ImportError(
            "This version of TensorFlow Addons requires TensorFlow "
            "version >= {required}; Detected an installation of version "
            "{present}. Please upgrade TensorFlow to proceed.".format(
                required=required_tensorflow_version, present=tf.__version__
            )
        )
