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
from tensorflow_addons.activations import lisht
from tensorflow_addons.activations.lisht import _lisht_py
from tensorflow_addons.utils import test_utils


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_theoretical_gradients(dtype):
    # Only test theoretical gradients for float32 and float64
    # because of the instability of float16 while computing jacobian
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)

    theoretical, numerical = tf.test.compute_gradient(lisht, [x])
    test_utils.assert_allclose_according_to_type(
        theoretical, numerical, rtol=5e-4, atol=5e-4
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_lisht(dtype):
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
    expected_result = tf.constant(
        [1.9280552, 0.7615942, 0.0, 0.7615942, 1.9280552], dtype=dtype
    )
    test_utils.assert_allclose_according_to_type(lisht(x), expected_result)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_same_as_py_func(dtype):
    np.random.seed(1234)
    for _ in range(20):
        verify_funcs_are_equivalent(dtype)


def verify_funcs_are_equivalent(dtype):
    x_np = np.random.uniform(-10, 10, size=(4, 4)).astype(dtype)
    x = tf.convert_to_tensor(x_np)

    with tf.GradientTape(persistent=True) as t:
        t.watch(x)
        y_native = lisht(x)
        y_py = _lisht_py(x)

    test_utils.assert_allclose_according_to_type(y_native, y_py)

    grad_native = t.gradient(y_native, x)
    grad_py = t.gradient(y_py, x)

    test_utils.assert_allclose_according_to_type(grad_native, grad_py)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
