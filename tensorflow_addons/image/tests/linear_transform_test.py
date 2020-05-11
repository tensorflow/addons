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
"""Test for linear transform."""
import pytest
import tensorflow as tf
import numpy as np
from tensorflow_addons.image import linear_transform


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_linear_transform():
    image = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    expected_output = tf.constant([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]])
    expected_output = expected_output.numpy()
    output = linear_transform(image)
    output = output.numpy()
    expected_output = np.resize(expected_output, (4, 3))
    output = np.resize(output, (4, 3))
    np.testing.assert_allclose(expected_output, output, 0.006)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_linear_transform_channels():
    image = tf.constant(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
            [[7, 7, 7], [8, 8, 8], [9, 9, 9]],
            [[10, 10, 10], [11, 11, 11], [12, 12, 12]],
        ]
    )
    expected_output = tf.constant(
        [
            [[2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6], [7, 7, 7]],
            [[8, 8, 8], [9, 9, 9], [10, 10, 10]],
            [[11, 11, 11], [12, 12, 12], [13, 13, 13]],
        ]
    )
    expected_output = expected_output.numpy()
    output = linear_transform(image)
    output = output.numpy()
    expected_output = np.resize(expected_output, (4, 3, 3))
    output = np.resize(output, (4, 3, 3))
    np.testing.assert_allclose(expected_output, output, 0.006)
