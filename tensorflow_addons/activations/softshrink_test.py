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

import sys

import pytest

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import softshrink
from tensorflow_addons.activations.softshrink import (
    _softshrink_py,
    _softshrink_custom_op,
)
from tensorflow_addons.utils import test_utils


def test_invalid():
    with pytest.raises(
        tf.errors.OpError, match="lower must be less than or equal to upper."
    ):
        y = _softshrink_custom_op(tf.ones(shape=(1, 2, 3)), lower=2.0, upper=-2.0)
        y.numpy()


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_softshrink(dtype):
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
    expected_result = tf.constant([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=dtype)
    test_utils.assert_allclose_according_to_type(softshrink(x), expected_result)

    expected_result = tf.constant([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=dtype)
    test_utils.assert_allclose_according_to_type(
        softshrink(x, lower=-1.0, upper=1.0), expected_result
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_same_as_py_func(dtype):
    np.random.seed(1234)
    for _ in range(20):
        verify_funcs_are_equivalent(dtype)


def verify_funcs_are_equivalent(dtype):
    x_np = np.random.uniform(-10, 10, size=(4, 4)).astype(dtype)
    x = tf.convert_to_tensor(x_np)
    lower = np.random.uniform(-10, 10)
    upper = lower + np.random.uniform(0, 10)

    with tf.GradientTape(persistent=True) as t:
        t.watch(x)
        y_native = softshrink(x, lower, upper)
        y_py = _softshrink_py(x, lower, upper)

    test_utils.assert_allclose_according_to_type(y_native, y_py)

    grad_native = t.gradient(y_native, x)
    grad_py = t.gradient(y_py, x)

    test_utils.assert_allclose_according_to_type(grad_native, grad_py)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_theoretical_gradients(dtype):
    # Only test theoretical gradients for float32 and float64
    # because of the instability of float16 while computing jacobian

    # Softshrink is not continuous at `lower` and `upper`.
    # Avoid these two points to make gradients smooth.
    x = tf.constant([-2.0, -1.5, 0.0, 1.5, 2.0], dtype=dtype)

    theoretical, numerical = tf.test.compute_gradient(softshrink, [x])
    test_utils.assert_allclose_according_to_type(theoretical, numerical, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
