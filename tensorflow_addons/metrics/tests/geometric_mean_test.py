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
"""Tests GeometricMean metrics."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.metrics import GeometricMean


def get_test_data():
    return [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
        ([0, 0, 0, 0, 0, 0, 0, 1, 2, 6], 0),
        ([0.2, 0.5, 0.3, 0.6, 0.1, 0.7], 0.32864603),
        ([8, 4, 1, 7, 2, 11, 9, 22, 52], 7.1804023),
        ([8.2, 9.7, 9.1, 2.7, 1.1, 2.0], 4.0324492),
        ([0.6666666, 0.215213, 0.15167], 0.27918512),
    ]


def assert_result(expected, result):
    np.testing.assert_allclose(expected, result, atol=1e-6)


def check_result(obj, expected_result, expected_count):
    result = obj.result().numpy()
    count = obj.count.numpy()
    assert_result(expected_result, result)
    np.testing.assert_equal(expected_count, count)


def test_config_gmean():
    def _check_config(obj, name):
        assert obj.name == name
        assert obj.dtype == tf.float32
        assert obj.stateful
        assert len(obj.variables) == 2

    name = "my_gmean"
    obj1 = GeometricMean(name=name)
    _check_config(obj1, name)

    obj2 = GeometricMean.from_config(obj1.get_config())
    _check_config(obj2, name)


def test_init_states_gmean():
    obj = GeometricMean()
    assert obj.total.numpy() == 0.0
    assert obj.count.numpy() == 0.0
    assert obj.total.dtype == tf.float32
    assert obj.count.dtype == tf.float32


@pytest.mark.parametrize("values, expected", get_test_data())
def test_scalar_update_state_gmean(values, expected):
    obj = GeometricMean()
    values = tf.constant(values, tf.float32)
    for v in values:
        obj.update_state(v)
    check_result(obj, expected, len(values))


@pytest.mark.parametrize("values, expected", get_test_data())
def test_vector_update_state_gmean(values, expected):
    obj = GeometricMean()
    values = tf.constant(values, tf.float32)
    obj.update_state(values)
    check_result(obj, expected, len(values))


@pytest.mark.parametrize("values, expected", get_test_data())
def test_call_gmean(values, expected):
    obj = GeometricMean()
    result = obj(tf.constant(values, tf.float32))
    count = obj.count.numpy()
    assert_result(expected, result)
    np.testing.assert_equal(len(values), count)


def test_reset_state():
    obj = GeometricMean()
    obj.update_state([1, 2, 3, 4, 5])
    obj.reset_state()
    assert obj.total.numpy() == 0.0
    assert obj.count.numpy() == 0.0


@pytest.mark.parametrize(
    "values, sample_weight, expected",
    [
        ([1, 2, 3, 4, 5], 1, 2.6051712),
        ([2.1, 4.6, 7.1], [1, 2, 3], 5.014777),
        ([9.6, 1.8, 8.2], [0.2, 0.5, 0.3], 3.9649222),
    ],
)
def test_sample_weight_gmean(values, sample_weight, expected):
    obj = GeometricMean()
    obj.update_state(values, sample_weight=sample_weight)
    assert_result(expected, obj.result().numpy())
