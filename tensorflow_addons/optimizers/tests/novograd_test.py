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
"""Tests for NovoGrad Optimizer."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.optimizers import NovoGrad


def run_dense_sample(iterations, expected, optimizer, dtype):
    var_0 = tf.Variable([1.0, 2.0], dtype=dtype)
    var_1 = tf.Variable([3.0, 4.0], dtype=dtype)

    grad_0 = tf.constant([0.1, 0.2], dtype=dtype)
    grad_1 = tf.constant([0.3, 0.4], dtype=dtype)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var_0.read_value(), expected[0], atol=2e-4)
    np.testing.assert_allclose(var_1.read_value(), expected[1], atol=2e-4)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dense_sample(dtype):
    run_dense_sample(
        iterations=1,
        expected=[[0.9552786425, 1.9105572849], [2.9400000012, 3.9200000016]],
        optimizer=NovoGrad(lr=0.1, epsilon=1e-8),
        dtype=dtype,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dense_sample_with_weight_decay(dtype):
    run_dense_sample(
        iterations=1,
        expected=[[0.945278642, 1.8905572849], [2.9100000012, 3.8800000016]],
        optimizer=NovoGrad(lr=0.1, weight_decay=0.1, epsilon=1e-8),
        dtype=dtype,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dense_sample_with_grad_averaging(dtype):
    run_dense_sample(
        iterations=2,
        expected=[[0.9105572849, 1.8211145698], [2.8800000024, 3.8400000032]],
        optimizer=NovoGrad(lr=0.1, grad_averaging=True, epsilon=1e-8),
        dtype=dtype,
    )


def run_sparse_sample(iterations, expected, optimizer, dtype):
    var_0 = tf.Variable([1.0, 2.0], dtype=dtype)
    var_1 = tf.Variable([3.0, 4.0], dtype=dtype)

    grad_0 = tf.IndexedSlices(
        tf.constant([0.1], dtype=dtype), tf.constant([0]), tf.constant([2])
    )
    grad_1 = tf.IndexedSlices(
        tf.constant([0.4], dtype=dtype), tf.constant([1]), tf.constant([2])
    )

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var_0.read_value(), expected[0])
    np.testing.assert_allclose(var_1.read_value(), expected[1])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparse_sample(dtype):
    run_sparse_sample(
        iterations=2,
        expected=[[0.71, 2.0], [3.0, 3.71]],
        optimizer=NovoGrad(lr=0.1, epsilon=1e-8),
        dtype=dtype,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparse_sample_with_weight_decay(dtype):
    run_sparse_sample(
        iterations=2,
        expected=[[0.6821, 2.0], [3.0, 3.5954]],
        optimizer=NovoGrad(lr=0.1, weight_decay=0.1, epsilon=1e-8),
        dtype=dtype,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparse_sample_with_grad_averaging(dtype):
    run_sparse_sample(
        iterations=2,
        expected=[[0.8, 2.0], [3.0, 3.8]],
        optimizer=NovoGrad(lr=0.1, grad_averaging=True, epsilon=1e-8),
        dtype=dtype,
    )


def test_fit_simple_linear_model():
    np.random.seed(0x2020)
    tf.random.set_seed(0x2020)

    x = np.random.standard_normal((100000, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((100000, 1)) * 1e-5

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
    model.compile(NovoGrad(), loss="mse")

    model.fit(x, y, epochs=2)

    x = np.random.standard_normal((100, 3))
    y = np.dot(x, w)
    predicted = model.predict(x)

    max_abs_diff = np.max(np.abs(predicted - y))
    assert max_abs_diff < 1e-2


def test_get_config():
    opt = NovoGrad(lr=1e-4, weight_decay=0.0, grad_averaging=False)
    config = opt.get_config()
    assert config["learning_rate"] == 1e-4
    assert config["weight_decay"] == 0.0
    assert config["grad_averaging"] is False


def test_serialization():
    optimizer = NovoGrad(lr=1e-4, weight_decay=0.0, grad_averaging=False)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
