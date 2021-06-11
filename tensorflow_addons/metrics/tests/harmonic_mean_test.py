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
"""Tests HarmonicMean metrics."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.metrics import HarmonicMean


def get_test_data():
    return [
        ([np.inf] * 2, 0),
        ([0, 0, 0, 0], 0),
        ([1, 4, 4], 2.0),
        ([0, 0, 0, 0, 0, 0, 0, 1, 2, 6], 0),
        ([0.2, 0.5, 0.3, 0.6, 0.1, 0.7], 0.25609756),
        ([8, 4, 1, 7, 2, 11, 9, 22, 52], 3.9394846),
        ([8.2, 9.7, 9.1, 2.7, 1.1, 2.0], 2.8376906),
        ([0.6666666, 0.215213, 0.15167], 0.23548213),
    ]


def assert_result(expected, result):
    np.testing.assert_allclose(expected, result, atol=1e-6)


def check_result(obj, expected_result, expected_count):
    result = obj.result().numpy()
    count = obj.count.numpy()
    assert_result(expected_result, result)
    np.testing.assert_equal(expected_count, count)


@pytest.mark.parametrize("values, expected", get_test_data())
def test_vector_update_state_hmean(values, expected):
    obj = HarmonicMean()
    values = tf.constant(values, tf.float32)
    obj.update_state(values)
    check_result(obj, expected, len(values))


@pytest.mark.parametrize("values, expected", get_test_data())
def test_call_hmean(values, expected):
    obj = HarmonicMean()
    result = obj(tf.constant(values, tf.float32))
    count = obj.count.numpy()
    assert_result(expected, result)
    np.testing.assert_equal(len(values), count)


@pytest.mark.parametrize(
    "values, sample_weight, expected",
    [
        ([1, 2, 3, 4, 5], 1, 2.1897807),
        ([2.1, 4.6, 7.1], [1, 2, 3], 4.499409),
        ([9.6, 1.8, 8.2], [0.2, 0.5, 0.3], 2.9833248),
    ],
)
def test_sample_weight_hmean(values, sample_weight, expected):
    obj = HarmonicMean()
    obj.update_state(values, sample_weight=sample_weight)
    assert_result(expected, obj.result().numpy())
