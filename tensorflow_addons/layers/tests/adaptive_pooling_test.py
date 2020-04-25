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
"""Tests for AdaptivePooling layers."""

import pytest
import numpy as np
from tensorflow_addons.layers.adaptive_pooling import (
    AdaptiveAveragePooling1D,
    AdaptiveMaxPooling1D,
    AdaptiveAveragePooling2D,
    AdaptiveMaxPooling2D,
    AdaptiveAveragePooling3D,
    AdaptiveMaxPooling3D,
)

from tensorflow_addons.utils import test_utils


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_avg_1d():
    valid_input = np.arange(start=0.0, stop=12.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 12, 1))
    output = np.array([1.0, 4.0, 7.0, 10.0]).astype(np.float32)
    output = np.reshape(output, (1, 4, 1))
    test_utils.layer_test(
        AdaptiveAveragePooling1D,
        kwargs={"output_size": 4, "data_format": "channels_last"},
        input_data=valid_input,
        expected_output=output,
    )

    valid_input = np.arange(start=0.0, stop=12.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 1, 12))
    output = np.array([1.0, 4.0, 7.0, 10.0]).astype(np.float32)
    output = np.reshape(output, (1, 1, 4))
    test_utils.layer_test(
        AdaptiveAveragePooling1D,
        kwargs={"output_size": 4, "data_format": "channels_first"},
        input_data=valid_input,
        expected_output=output,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_avg_2d():
    valid_input = np.arange(start=0.0, stop=40.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 10, 1))
    output = np.array([[7.0, 12.0], [27.0, 32.0]]).astype(np.float32)
    output = np.reshape(output, (1, 2, 2, 1))
    test_utils.layer_test(
        AdaptiveAveragePooling2D,
        kwargs={"output_size": (2, 2), "data_format": "channels_last"},
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


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_avg_3d():
    valid_input = np.arange(start=0.0, stop=80.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 10, 2, 1))
    output = np.array(
        [[[14.0, 15.0], [24.0, 25.0]], [[54.0, 55.0], [64.0, 65.0]]]
    ).astype(np.float32)
    output = np.reshape(output, (1, 2, 2, 2, 1))
    test_utils.layer_test(
        AdaptiveAveragePooling3D,
        kwargs={"output_size": (2, 2, 2), "data_format": "channels_last"},
        input_data=valid_input,
        expected_output=output,
    )

    valid_input = np.arange(start=0.0, stop=80.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 1, 4, 10, 2))
    output = np.array(
        [[[14.0, 15.0], [24.0, 25.0]], [[54.0, 55.0], [64.0, 65.0]]]
    ).astype(np.float32)
    output = np.reshape(output, (1, 1, 2, 2, 2))
    test_utils.layer_test(
        AdaptiveAveragePooling3D,
        kwargs={"output_size": (2, 2, 2), "data_format": "channels_first"},
        input_data=valid_input,
        expected_output=output,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_max_1d():
    valid_input = np.arange(start=0.0, stop=12.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 12, 1))
    output = np.array([2.0, 5.0, 8.0, 11.0]).astype(np.float32)
    output = np.reshape(output, (1, 4, 1))
    test_utils.layer_test(
        AdaptiveMaxPooling1D,
        kwargs={"output_size": 4, "data_format": "channels_last"},
        input_data=valid_input,
        expected_output=output,
    )

    valid_input = np.arange(start=0.0, stop=12.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 1, 12))
    output = np.array([2.0, 5.0, 8.0, 11.0]).astype(np.float32)
    output = np.reshape(output, (1, 1, 4))
    test_utils.layer_test(
        AdaptiveMaxPooling1D,
        kwargs={"output_size": 4, "data_format": "channels_first"},
        input_data=valid_input,
        expected_output=output,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_max_2d():
    valid_input = np.arange(start=0.0, stop=40.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 10, 1))
    output = np.array([[14.0, 19.0], [34.0, 39.0]]).astype(np.float32)
    output = np.reshape(output, (1, 2, 2, 1))
    test_utils.layer_test(
        AdaptiveMaxPooling2D,
        kwargs={"output_size": (2, 2), "data_format": "channels_last"},
        input_data=valid_input,
        expected_output=output,
    )

    valid_input = np.arange(start=0.0, stop=40.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 1, 4, 10))
    output = np.array([[14.0, 19.0], [34.0, 39.0]]).astype(np.float32)
    output = np.reshape(output, (1, 1, 2, 2))
    test_utils.layer_test(
        AdaptiveMaxPooling2D,
        kwargs={"output_size": (2, 2), "data_format": "channels_first"},
        input_data=valid_input,
        expected_output=output,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_max_3d():
    valid_input = np.arange(start=0.0, stop=80.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 10, 2, 1))
    output = np.array(
        [[[28.0, 29.0], [38.0, 39.0]], [[68.0, 69.0], [78.0, 79.0]]]
    ).astype(np.float32)
    output = np.reshape(output, (1, 2, 2, 2, 1))
    test_utils.layer_test(
        AdaptiveMaxPooling3D,
        kwargs={"output_size": (2, 2, 2), "data_format": "channels_last"},
        input_data=valid_input,
        expected_output=output,
    )

    valid_input = np.arange(start=0.0, stop=80.0, step=1.0).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 1, 4, 10, 2))
    output = np.array(
        [[[28.0, 29.0], [38.0, 39.0]], [[68.0, 69.0], [78.0, 79.0]]]
    ).astype(np.float32)
    output = np.reshape(output, (1, 1, 2, 2, 2))
    test_utils.layer_test(
        AdaptiveMaxPooling3D,
        kwargs={"output_size": (2, 2, 2), "data_format": "channels_first"},
        input_data=valid_input,
        expected_output=output,
    )
