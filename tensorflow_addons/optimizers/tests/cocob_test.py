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
"""Tests for COntinuos COin Betting (COCOB) Backprop optimizer"""

import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import COCOB


def run_dense_sample(iterations, expected, optimizer):
    var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
    var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

    grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
    grad_1 = tf.constant([0.03, 0.04], dtype=tf.dtypes.float32)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var_0.read_value(), expected[0], atol=2e-4)
    np.testing.assert_allclose(var_1.read_value(), expected[1], atol=2e-4)


def test_dense_sample_with_default_alpha():
    run_dense_sample(
        iterations=10,
        expected=[[0.84528893, 1.845289], [2.845289, 3.845289]],
        optimizer=COCOB(),
    )


def test_dense_sample_with_custom_int_alpha():
    run_dense_sample(
        iterations=7,
        expected=[[0.09346926, 1.0934693], [2.0934694, 3.0934694]],
        optimizer=COCOB(20),
    )


def test_dense_sample_with_custom_float_alpha():
    run_dense_sample(
        iterations=5,
        expected=[[0.89307845, 1.8930784], [2.8930783, 3.8930783]],
        optimizer=COCOB(55.7),
    )
