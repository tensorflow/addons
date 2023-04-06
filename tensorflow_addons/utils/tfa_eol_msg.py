# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import warnings


def _print_eol_warning():
    """
    Prints TensorFlow Addons End of Life Warning
    """
    warnings.warn(
        "\n\nTensorFlow Addons (TFA) has ended development and introduction of new features.\n"
        "TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.\n"
        "Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community "
        "(e.g. Keras, Keras-CV, and Keras-NLP). \n\n"
        "For more information see: https://github.com/tensorflow/addons/issues/2807 \n",
        UserWarning,
    )


_print_eol_warning()
