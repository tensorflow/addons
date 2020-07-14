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
"""Tests for Proximal Adagrad optimizer."""

import numpy as np
import tensorflow as tf

from tensorflow_addons.optimizers import ProximalAdagrad


def run_sample(iterations, expected, optimizer, sparse=False, rtol=1e-7, atol=0.0):
    var_0 = tf.Variable([1.0, 2.0], dtype=tf.float32)
    var_1 = tf.Variable([4.0, 3.0], dtype=tf.float32)

    if sparse:
        grad_0 = tf.IndexedSlices(
            tf.constant([0.1], dtype=tf.float32), tf.constant([0]), tf.constant([2])
        )
        grad_1 = tf.IndexedSlices(
            tf.constant([0.02], dtype=tf.float32), tf.constant([1]), tf.constant([2])
        )
    else:
        grad_0 = tf.constant([0.1, 0.2], dtype=tf.float32)
        grad_1 = tf.constant([0.01, 0.02], dtype=tf.float32)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var_0.read_value(), expected[0], rtol, atol)
    np.testing.assert_allclose(var_1.read_value(), expected[1], rtol, atol)


def test_without_regularization():
    run_sample(
        iterations=10,
        expected=[[-6.722771, -9.230448], [3.0539124, 1.1230775]],
        optimizer=ProximalAdagrad(lr=3.0, initial_accumulator_value=0.1),
    )


def test_with_l1_regularization():
    run_sample(
        iterations=10,
        expected=[[-6.663634, -9.190331], [2.9593036, 1.0292315]],
        optimizer=ProximalAdagrad(
            lr=3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001
        ),
    )


def test_with_l1_l2_regularization():
    run_sample(
        iterations=10,
        expected=[[-0.0495, -0.0995], [-0.0045, -0.0095]],
        optimizer=ProximalAdagrad(
            lr=3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0,
        ),
    )


def test_sparse_without_regularization():
    run_sample(
        iterations=10,
        expected=[[-6.722771, 2.0], [4.0, 1.1230775]],
        optimizer=ProximalAdagrad(lr=3.0, initial_accumulator_value=0.1),
        sparse=True,
        rtol=5e-7,
    )


def test_sparse_with_l1_regularization():
    run_sample(
        iterations=10,
        expected=[[-6.663634, 2.0], [4.0, 1.0292315]],
        optimizer=ProximalAdagrad(
            lr=3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001
        ),
        sparse=True,
        rtol=5e-7,
    )


def test_sparse_with_l1_l2_regularization():
    run_sample(
        iterations=10,
        expected=[[-0.0495, 2.0], [4.0, -0.0095]],
        optimizer=ProximalAdagrad(
            lr=3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0,
        ),
        sparse=True,
    )


def test_serialization():
    optimizer = ProximalAdagrad(
        lr=1e-4,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.1,
        l2_regularization_strength=0.1,
    )
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
