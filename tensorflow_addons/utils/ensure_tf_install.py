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

from distutils.version import LooseVersion
import warnings

import tensorflow as tf


warning_template = """
This version of TensorFlow Addons requires TensorFlow {required}.
Detected an installation of version {present}.

While some functions might work, TensorFlow Addons was not tested
with this TensorFlow version. Also custom ops were not compiled
against this version of TensorFlow. If you use custom ops,
you might get errors (segmentation faults for example).

It might help you to fallback to pure Python ops with
TF_ADDONS_PY_OPS . To do that, see
https://github.com/tensorflow/addons#gpucpu-custom-ops

If you encounter errors, do *not* file bugs in GitHub because
the version of TensorFlow you are using is not supported.
"""


def _ensure_tf_install():
    """Warn the user if the version of TensorFlow used is not supported.
    """

    # Update this whenever we need to depend on a newer TensorFlow release.
    required_tf_version = "2.1.0"

    if LooseVersion(tf.__version__) != LooseVersion(required_tf_version):
        message = warning_template.format(
            required=required_tf_version, present=tf.__version__
        )
        warnings.warn(message, UserWarning)
