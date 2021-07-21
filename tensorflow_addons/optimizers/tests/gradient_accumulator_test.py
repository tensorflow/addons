# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for GradientAccumulator optimizers."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.optimizers import GradientAccumulator


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu", tf.distribute.MirroredStrategy])
def test_run():
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0, 4.0])
    accum_steps = 4

    grads0 = tf.constant([0.1, 0.1])
    grads1 = tf.constant([0.01, 0.01])

    grads_and_vars = list(zip([grads0, grads1], [var0, var1]))

    opt = GradientAccumulator(tf.keras.optimizers.SGD(lr=1.0), accum_steps)

    strategy = tf.distribute.get_strategy()
    for _ in range(accum_steps + 1):
        strategy.run(opt.apply_gradients, [grads_and_vars])

    np.testing.assert_allclose(var0.read_value(), [0.6, 1.6])
    np.testing.assert_allclose(var1.read_value(), [2.96, 3.96])
    np.testing.assert_allclose(opt.iterations.read_value(), 1)
    np.testing.assert_allclose(opt.step.read_value(), accum_steps + 1)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu", tf.distribute.MirroredStrategy])
def test_sparse():
    var0 = tf.Variable([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    var1 = tf.Variable([[3.0, 4.0, 0.0]])

    grads0 = tf.IndexedSlices(
        tf.constant([[0.1, 0.1, 0.0]]),
        tf.constant([1]),
        tf.constant([1, 3]),
    )
    grads1 = tf.IndexedSlices(
        tf.constant([[0.01, 0.01, 0.0]]),
        tf.constant([0]),
        tf.constant([1, 3]),
    )

    grads_and_vars = list(zip([grads0, grads1], [var0, var1]))
    opt = GradientAccumulator(tf.keras.optimizers.SGD(lr=1.0))
    strategy = tf.distribute.get_strategy()
    for _ in range(8):
        strategy.run(opt.apply_gradients, [grads_and_vars])
    np.testing.assert_allclose(var0.read_value(), [[1.0, 2.0, 0.0], [0.2, 1.2, 0.0]])
    np.testing.assert_allclose(var1.read_value(), [[2.92, 3.92, 0.0]])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense():
    grad = tf.Variable([[0.1]])
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.keras.initializers.Constant([[1.0]]),
                use_bias=False,
            )
        ]
    )
    model.build(input_shape=[1, 1])

    opt = GradientAccumulator(tf.keras.optimizers.SGD(lr=2.0), accum_steps=2)
    _ = opt.apply_gradients(list(zip([grad], model.variables)))
    np.testing.assert_allclose(model.variables[0].read_value(), [[1.0]])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_optimizer_string():
    _ = GradientAccumulator("adam")


def test_config():
    sgd_opt = tf.keras.optimizers.SGD(lr=2.0, nesterov=True, momentum=0.3, decay=0.1)
    accum_steps = 4
    opt = GradientAccumulator(sgd_opt, accum_steps=accum_steps)
    config = opt.get_config()

    assert config["accum_steps"] == accum_steps

    new_opt = GradientAccumulator.from_config(config)
    old_sgd_config = opt._optimizer.get_config()
    new_sgd_config = new_opt._optimizer.get_config()

    for k1, k2 in zip(old_sgd_config, new_sgd_config):
        assert old_sgd_config[k1] == new_sgd_config[k2]


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device([tf.distribute.MirroredStrategy])
def test_fit_simple_linear_model():
    seed = 0x2019
    np.random.seed(seed)
    tf.random.set_seed(seed)
    num_examples = 5000
    x = np.random.standard_normal((num_examples, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((num_examples, 1)) * 1e-4

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))

    opt = GradientAccumulator("sgd")
    model.compile(opt, loss="mse")

    model.fit(x, y, epochs=5)

    x = np.random.standard_normal((100, 3))
    y = np.dot(x, w)

    predicted = model.predict(x)

    max_abs_diff = np.max(np.abs(predicted - y))
    assert max_abs_diff < 5e-3


def test_serialization():
    sgd_opt = tf.keras.optimizers.SGD(lr=2.0, nesterov=True, momentum=0.3, decay=0.1)
    optimizer = GradientAccumulator(sgd_opt)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_model_mixed_precision():
    x = np.random.standard_normal((10000, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((10000, 1)) * 1e-4
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
    model.compile(GradientAccumulator("sgd"), loss="mse")
    model.fit(x, y, epochs=3)
