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
"""Tests for Lookahead optimizer."""

import os

import numpy as np
import pytest
import tensorflow as tf
import tempfile

from tensorflow_addons.optimizers import Lookahead
from tensorflow_addons.utils import test_utils
from distutils.version import LooseVersion


def run_dense_sample(iterations, optimizer, seed=0x2019):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    val_0 = np.random.random((2,))
    val_1 = np.random.random((2,))

    var_0 = tf.Variable(val_0, dtype=tf.dtypes.float32)
    var_1 = tf.Variable(val_1, dtype=tf.dtypes.float32)

    grad_0 = tf.constant(np.random.standard_normal((2,)), dtype=tf.dtypes.float32)
    grad_1 = tf.constant(np.random.standard_normal((2,)), dtype=tf.dtypes.float32)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)

    return [val_0, val_1], [var_0, var_1]


def run_sparse_sample(iterations, optimizer, seed=0x2019):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    val_0 = np.random.random((2,))
    val_1 = np.random.random((2,))

    var_0 = tf.Variable(val_0, dtype=tf.dtypes.float32)
    var_1 = tf.Variable(val_1, dtype=tf.dtypes.float32)

    grad_0 = tf.IndexedSlices(
        tf.constant([np.random.standard_normal()]), tf.constant([0]), tf.constant([2]),
    )
    grad_1 = tf.IndexedSlices(
        tf.constant([np.random.standard_normal()]), tf.constant([1]), tf.constant([2]),
    )

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)

    return [val_0, val_1], [var_0, var_1]


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_exact_ratio():
    for k in [5, 10, 100]:
        for alpha in [0.3, 0.7]:
            optimizer = tf.keras.optimizers.get("adam")
            vals, quick_vars = run_dense_sample(k, optimizer)
            optimizer = Lookahead("adam", sync_period=k, slow_step_size=alpha)
            _, slow_vars = run_dense_sample(k, optimizer)
            for val, quick, slow in zip(vals, quick_vars, slow_vars):
                expected = val + (quick - val) * alpha
                np.testing.assert_allclose(
                    expected.numpy(), slow.numpy(), rtol=1e-06, atol=1e-06
                )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_exact_ratio():
    for k in [5, 10, 100]:
        for alpha in [0.3, 0.7]:
            optimizer = tf.keras.optimizers.get("adam")
            vals, quick_vars = run_sparse_sample(k, optimizer)
            optimizer = Lookahead("adam", sync_period=k, slow_step_size=alpha)
            _, slow_vars = run_sparse_sample(k, optimizer)
            for val, quick, slow in zip(vals, quick_vars, slow_vars):
                expected = val + (quick - val) * alpha
                np.testing.assert_allclose(
                    expected.numpy(), slow.numpy(), rtol=1e-06, atol=1e-06
                )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_fit_simple_linear_model():
    np.random.seed(0x2019)
    tf.random.set_seed(0x2019)

    x = np.random.standard_normal((10000, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((10000, 1)) * 1e-4

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
    model.compile(Lookahead("sgd"), loss="mse")

    model.fit(x, y, epochs=3)

    x = np.random.standard_normal((100, 3))
    y = np.dot(x, w)
    predicted = model.predict(x)

    max_abs_diff = np.max(np.abs(predicted - y))
    assert max_abs_diff < 1e-3


def test_fit_simple_linear_model_mixed_precision():
    if test_utils.is_gpu_available() and LooseVersion(tf.__version__) <= "2.2.0":
        pytest.xfail("See https://github.com/tensorflow/tensorflow/issues/39775")
    np.random.seed(0x2019)
    tf.random.set_seed(0x2019)

    x = np.random.standard_normal((10000, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((10000, 1)) * 1e-4

    try:
        tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
        model.compile(Lookahead("sgd"), loss="mse")
    finally:
        tf.keras.mixed_precision.experimental.set_policy("float32")
    model.fit(x, y, epochs=3)

    x = np.random.standard_normal((100, 3))
    y = np.dot(x, w)
    predicted = model.predict(x)

    max_abs_diff = np.max(np.abs(predicted - y))
    assert max_abs_diff < 2.3e-3
    assert max_abs_diff >= 1e-3


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_model_dynamic_lr():
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

    opt = Lookahead("adam", sync_period=10, slow_step_size=0.4)
    _ = opt.apply_gradients(list(zip([grad], model.variables)))

    np.testing.assert_allclose(opt.lr.read_value(), 1e-3)

    opt.lr = 1e-4
    np.testing.assert_allclose(opt.lr.read_value(), 1e-4)


def test_get_config():
    opt = Lookahead("adam", sync_period=10, slow_step_size=0.4)
    opt = tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(opt))
    config = opt.get_config()
    assert config["sync_period"] == 10
    assert config["slow_step_size"] == 0.4


def test_serialization():
    optimizer = Lookahead("adam", sync_period=10, slow_step_size=0.4)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()


def _init_model(optimizer, init_w):
    model = tf.keras.models.Sequential()
    dense = tf.keras.layers.Dense(input_shape=(3,), units=1)
    model.add(dense)
    model.compile(Lookahead(optimizer), loss="mse")
    dense.set_weights([init_w, np.zeros(1,)])
    return model


def assert_same_optimizer_states(optimizer, new_optimizer):
    # Remove the iteration variable
    weights = []
    for weight in optimizer.weights:
        if "iter" not in weight.name:
            weights.append(weight)
    new_weights = []
    for weight in new_optimizer.weights:
        if "iter" not in weight.name:
            new_weights.append(weight)

    assert len(weights) == len(new_weights)

    weights = sorted(weights, key=lambda w: w.name)
    new_weights = sorted(new_weights, key=lambda w: w.name)

    for weight, new_weight in zip(weights, new_weights):
        assert np.allclose(weight.numpy(), new_weight.numpy(), atol=1e-4)

    # Assert recursively
    if hasattr(optimizer, "_optimizer"):
        assert_same_optimizer_states(optimizer._optimizer, new_optimizer._optimizer)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
@pytest.mark.parametrize("weights_only", [False, True])
def test_save_load(optimizer, weights_only):
    x = np.random.standard_normal((10000, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((10000, 1)) * 1e-4

    init_w = np.random.standard_normal((3, 1))

    model = _init_model(optimizer, init_w)
    model.fit(x, y, epochs=2, shuffle=False)

    with tempfile.TemporaryDirectory() as ckpt_dir:
        new_model = _init_model(optimizer, init_w)
        new_model.fit(x, y, epochs=1, shuffle=False)

        ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
        if weights_only:
            new_model.save_weights(ckpt_path)
            new_model = _init_model(optimizer, init_w)
            new_model.load_weights(ckpt_path)
        else:
            new_model.save(ckpt_path)
            new_model = tf.keras.models.load_model(
                ckpt_path,
                custom_objects={
                    "Lookahead": Lookahead,
                }
            )

        new_model.fit(x, y, epochs=1, shuffle=False)

        assert_same_optimizer_states(model.optimizer, new_model.optimizer)
