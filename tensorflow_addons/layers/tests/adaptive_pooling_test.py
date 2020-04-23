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
"""Tests for AdaptivePooling2D layer."""

import pytest
import numpy as np
from tensorflow_addons.layers.adaptive_pooling import AdaptiveAveragePooling2D

from tensorflow_addons.utils import test_utils


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_simple():
    valid_input = np.arange(start=0.0, stop=40.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 10, 1))
    output = np.array([[7.0, 12.0], [27.0, 32.0]]).astype(np.float32)
    output = np.reshape(output, (1, 2, 2, 1))
    test_utils.layer_test(
        AdaptiveAveragePooling2D,
        kwargs={"output_size": (2, 2)},
        input_data=valid_input,
        expected_output=output,
    )

    valid_input = np.arange(start=0.0, stop=40.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 1, 4, 10))
    output = np.array([[7.0, 12.0], [27.0, 32.0]]).astype(np.float32)
    output = np.reshape(output, (1, 1, 2, 2))
    test_utils.layer_test(
        AdaptiveAveragePooling2D,
        kwargs={"output_size": (2, 2), "data_format": "channels_first"},
        input_data=valid_input,
        expected_output=output,
    )
