# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Stochastic Weight Averaging optimizer."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.optimizers import stochastic_weight_averaging
from tensorflow_addons.optimizers.utils import fit_bn

SWA = stochastic_weight_averaging.SWA


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_averaging():
    start_averaging = 0
    average_period = 1
    if hasattr(tf.keras.optimizers, "legacy"):
        sgd = tf.keras.optimizers.legacy.SGD(lr=1.0)
    else:
        sgd = tf.keras.optimizers.SGD(lr=1.0)
    optimizer = SWA(sgd, start_averaging, average_period)

    val_0 = [1.0, 1.0]
    val_1 = [2.0, 2.0]
    var_0 = tf.Variable(val_0)
    var_1 = tf.Variable(val_1)

    grad_val_0 = [0.1, 0.1]
    grad_val_1 = [0.1, 0.1]
    grad_0 = tf.constant(grad_val_0)
    grad_1 = tf.constant(grad_val_1)
    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    optimizer.apply_gradients(grads_and_vars)
    optimizer.apply_gradients(grads_and_vars)
    optimizer.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var_1.read_value(), [1.7, 1.7], rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(var_0.read_value(), [0.7, 0.7], rtol=1e-06, atol=1e-06)

    optimizer.assign_average_vars([var_0, var_1])

    np.testing.assert_allclose(var_0.read_value(), [0.8, 0.8])
    np.testing.assert_allclose(var_1.read_value(), [1.8, 1.8])


def test_optimizer_failure():
    with pytest.raises(TypeError):
        _ = SWA(None, average_period=10)


def test_optimizer_string():
    _ = SWA("adam", average_period=10)


def test_get_config():
    opt = SWA("adam", average_period=10, start_averaging=0)
    opt = tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(opt))
    config = opt.get_config()
    assert config["average_period"] == 10
    assert config["start_averaging"] == 0


def test_assign_batchnorm():
    x = np.random.standard_normal((10, 64))
    y = np.random.standard_normal((10, 1))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1))

    if hasattr(tf.keras.optimizers, "legacy"):
        opt = SWA(tf.keras.optimizers.legacy.SGD())
    else:
        opt = SWA(tf.keras.optimizers.SGD())
    model.compile(optimizer=opt, loss="mean_squared_error")
    model.fit(x, y, epochs=1)

    opt.assign_average_vars(model.variables)
    fit_bn(model, x, y)


def test_fit_simple_linear_model():
    seed = 0x2019
    np.random.seed(seed)
    tf.random.set_seed(seed)
    num_examples = 100000
    x = np.random.standard_normal((num_examples, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((num_examples, 1)) * 1e-4

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
    # using num_examples - 1 since steps starts from 0.
    optimizer = SWA("sgd", start_averaging=num_examples // 32 - 1, average_period=100)
    model.compile(optimizer, loss="mse")
    model.fit(x, y, epochs=2)
    optimizer.assign_average_vars(model.variables)

    x = np.random.standard_normal((100, 3))
    y = np.dot(x, w)

    predicted = model.predict(x)

    max_abs_diff = np.max(np.abs(predicted - y))
    assert max_abs_diff < 1e-3


def test_serialization():
    start_averaging = 0
    average_period = 1
    if hasattr(tf.keras.optimizers, "legacy"):
        sgd = tf.keras.optimizers.legacy.SGD(lr=1.0)
    else:
        sgd = tf.keras.optimizers.SGD(lr=1.0)
    optimizer = SWA(sgd, start_averaging, average_period)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
