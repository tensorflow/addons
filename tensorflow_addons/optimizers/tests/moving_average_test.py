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
"""Tests for MovingAverage optimizers."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.optimizers import MovingAverage


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_run():
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0, 4.0])

    grads0 = tf.constant([0.1, 0.1])
    grads1 = tf.constant([0.01, 0.01])

    grads_and_vars = list(zip([grads0, grads1], [var0, var1]))

    if hasattr(tf.keras.optimizers, "legacy"):
        opt = MovingAverage(tf.keras.optimizers.legacy.SGD(lr=2.0), average_decay=0.5)
    else:
        opt = MovingAverage(tf.keras.optimizers.SGD(lr=2.0), average_decay=0.5)

    opt.apply_gradients(grads_and_vars)
    opt.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var0.read_value(), [0.6, 1.6])
    np.testing.assert_allclose(var1.read_value(), [2.96, 3.96])

    ema_var0 = opt.get_slot(var0, "average")
    ema_var1 = opt.get_slot(var1, "average")

    np.testing.assert_allclose(ema_var0.read_value(), [0.75, 1.75])
    np.testing.assert_allclose(ema_var1.read_value(), [2.975, 3.975])

    _ = opt.assign_average_vars([var0, var1])

    np.testing.assert_allclose(var0.read_value(), [0.75, 1.75])
    np.testing.assert_allclose(var1.read_value(), [2.975, 3.975])

    var0.assign_add([1.0, 1.0]),
    var1.assign_add([2.0, 2.0]),
    ema_var0.assign_add([3.0, 3.0]),
    ema_var1.assign_add([4.0, 4.0]),

    np.testing.assert_allclose(var0.read_value(), [1.75, 2.75])
    np.testing.assert_allclose(var1.read_value(), [4.975, 5.975])
    np.testing.assert_allclose(ema_var0.read_value(), [3.75, 4.75])
    np.testing.assert_allclose(ema_var1.read_value(), [6.975, 7.975])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_opt_failure():
    base_opt = None
    with pytest.raises(TypeError):
        MovingAverage(base_opt, 0.5)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_num_updates_valid():
    for num_updates in [1, tf.Variable(1)]:
        MovingAverage("sgd", num_updates=num_updates)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_num_updates_invalid():
    for num_updates in [1.0, tf.Variable(1.0), "a"]:
        with pytest.raises(TypeError):
            MovingAverage("sgd", num_updates=num_updates)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_model_weights_update():
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

    if hasattr(tf.keras.optimizers, "legacy"):
        opt = MovingAverage(tf.keras.optimizers.legacy.SGD(lr=2.0), average_decay=0.5)
    else:
        opt = MovingAverage(tf.keras.optimizers.SGD(lr=2.0), average_decay=0.5)
    _ = opt.apply_gradients(list(zip([grad], model.variables)))
    np.testing.assert_allclose(model.variables[0].read_value(), [[0.8]])
    _ = opt.assign_average_vars(model.variables)
    np.testing.assert_allclose(model.variables[0].read_value(), [[0.9]])


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
    if hasattr(tf.keras.optimizers, "legacy"):
        opt = MovingAverage(tf.keras.optimizers.legacy.SGD(lr=1e-3), average_decay=0.5)
    else:
        opt = MovingAverage(tf.keras.optimizers.SGD(lr=1e-3), average_decay=0.5)
    _ = opt.apply_gradients(list(zip([grad], model.variables)))
    np.testing.assert_allclose(opt.lr.read_value(), 1e-3)
    opt.lr = 1e-4
    np.testing.assert_allclose(opt.lr.read_value(), 1e-4)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_optimizer_string():
    _ = MovingAverage("adam")


def test_config():
    if hasattr(tf.keras.optimizers, "legacy"):
        sgd_opt = tf.keras.optimizers.legacy.SGD(
            lr=2.0, nesterov=True, momentum=0.3, decay=0.1
        )
    else:
        sgd_opt = tf.keras.optimizers.SGD(
            lr=2.0, nesterov=True, momentum=0.3, decay=0.1
        )
    opt = MovingAverage(
        sgd_opt,
        average_decay=0.5,
        num_updates=None,
        start_step=5,
        dynamic_decay=True,
    )
    config = opt.get_config()

    assert config["average_decay"] == 0.5
    assert config["num_updates"] is None
    assert config["start_step"] == 5
    assert config["dynamic_decay"] is True

    new_opt = MovingAverage.from_config(config)
    old_sgd_config = opt._optimizer.get_config()
    new_sgd_config = new_opt._optimizer.get_config()

    for k1, k2 in zip(old_sgd_config, new_sgd_config):
        assert old_sgd_config[k1] == new_sgd_config[k2]


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
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

    opt = MovingAverage("sgd")
    model.compile(opt, loss="mse")

    model.fit(x, y, epochs=5)
    opt.assign_average_vars(model.variables)

    x = np.random.standard_normal((100, 3))
    y = np.dot(x, w)

    predicted = model.predict(x)

    max_abs_diff = np.max(np.abs(predicted - y))
    assert max_abs_diff < 5e-3


def test_serialization():
    if hasattr(tf.keras.optimizers, "legacy"):
        sgd_opt = tf.keras.optimizers.legacy.SGD(
            lr=2.0, nesterov=True, momentum=0.3, decay=0.1
        )
    else:
        sgd_opt = tf.keras.optimizers.SGD(
            lr=2.0, nesterov=True, momentum=0.3, decay=0.1
        )
    optimizer = MovingAverage(
        sgd_opt,
        average_decay=0.5,
        num_updates=None,
        start_step=5,
        dynamic_decay=True,
    )
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_start_step():
    var0 = tf.Variable([1.0, 2.0])
    grads0 = tf.constant([0.1, 0.1])
    grads_and_vars = [(grads0, var0)]

    if hasattr(tf.keras.optimizers, "legacy"):
        sgd_opt = tf.keras.optimizers.legacy.SGD(lr=1.0)
    else:
        sgd_opt = tf.keras.optimizers.SGD(lr=1.0)

    opt = MovingAverage(sgd_opt, average_decay=0.5, start_step=1)

    opt.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var0.read_value(), [0.9, 1.9])

    ema_var0 = opt.get_slot(var0, "average")

    opt.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var0.read_value(), [0.8, 1.8])

    np.testing.assert_allclose(ema_var0.read_value(), [0.85, 1.85])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dynamic_decay():
    var0 = tf.Variable([1.0, 2.0])
    grads0 = tf.constant([0.1, 0.1])
    grads_and_vars = [(grads0, var0)]

    if hasattr(tf.keras.optimizers, "legacy"):
        opt = MovingAverage(
            tf.keras.optimizers.legacy.SGD(lr=2.0),
            average_decay=0.5,
            dynamic_decay=True,
        )
    else:
        opt = MovingAverage(
            tf.keras.optimizers.SGD(lr=2.0),
            average_decay=0.5,
            dynamic_decay=True,
        )

    opt.apply_gradients(grads_and_vars)
    opt.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var0.read_value(), [0.6, 1.6])

    ema_var0 = opt.get_slot(var0, "average")
    np.testing.assert_allclose(ema_var0.read_value(), [0.64, 1.64])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device([tf.distribute.MirroredStrategy])
def test_swap_weight_no_shadow_copy(device):
    with device.scope():
        var = tf.Variable([1.0, 2.0])
        grads = tf.constant([0.1, 0.1])

        if hasattr(tf.keras.optimizers, "legacy"):
            opt = MovingAverage(
                tf.keras.optimizers.legacy.SGD(lr=2.0), average_decay=0.5
            )
        else:
            opt = MovingAverage(tf.keras.optimizers.SGD(lr=2.0), average_decay=0.5)

    @tf.function
    def apply_gradients():
        opt.apply_gradients([(grads, var)])

    device.run(apply_gradients)

    np.testing.assert_allclose(var.read_value(), [0.8, 1.8])
    ema_var = opt.get_slot(var, "average")
    np.testing.assert_allclose(ema_var.read_value(), [0.9, 1.9])

    with device.scope():
        opt.swap_weights()

    np.testing.assert_allclose(ema_var.read_value(), [0.8, 1.8])
    np.testing.assert_allclose(var.read_value(), [0.9, 1.9])

    with device.scope():
        opt.swap_weights()

    np.testing.assert_allclose(var.read_value(), [0.8, 1.8])
    np.testing.assert_allclose(ema_var.read_value(), [0.9, 1.9])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device([tf.distribute.MirroredStrategy])
def test_swap_weights(device):
    with device.scope():
        var = tf.Variable([1.0, 2.0])
        grads = tf.constant([0.1, 0.1])

        if hasattr(tf.keras.optimizers, "legacy"):
            opt = MovingAverage(
                tf.keras.optimizers.legacy.SGD(lr=2.0), average_decay=0.5
            )
        else:
            opt = MovingAverage(tf.keras.optimizers.SGD(lr=2.0), average_decay=0.5)

    @tf.function
    def apply_gradients():
        opt.apply_gradients([(grads, var)])

    device.run(apply_gradients)

    np.testing.assert_allclose(var.read_value(), [0.8, 1.8])
    ema_var = opt.get_slot(var, "average")
    np.testing.assert_allclose(ema_var.read_value(), [0.9, 1.9])

    with device.scope():
        opt.shadow_copy([var])
        opt.swap_weights()

    np.testing.assert_allclose(ema_var.read_value(), [0.8, 1.8])
    np.testing.assert_allclose(var.read_value(), [0.9, 1.9])

    with device.scope():
        opt.swap_weights()

    np.testing.assert_allclose(var.read_value(), [0.8, 1.8])
    np.testing.assert_allclose(ema_var.read_value(), [0.9, 1.9])


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_model_mixed_precision():
    x = np.random.standard_normal((10000, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((10000, 1)) * 1e-4
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
    model.compile(MovingAverage("sgd"), loss="mse")
    model.fit(x, y, epochs=3)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_no_average_slot():
    max_features = 5000
    max_len = 4
    embedding_dims = 2

    # Some preprocessing layers have TrackableWeightHandler.
    # They are returned when using model.variables
    # but it's unable to assign average slot to them.
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=max_len,
    )

    vectorize_layer.adapt(["foo", "bar", "baz"])

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(1,), dtype=tf.string),
            vectorize_layer,
            tf.keras.layers.Embedding(max_features + 1, embedding_dims),
            tf.keras.layers.Dense(1),
        ]
    )

    optimizer = MovingAverage("sgd")

    model.compile(optimizer, loss="mse")
    model.fit(x=["foo", "bar", "baz"], y=[0.0, 1.0, 2.0], epochs=1)

    optimizer.assign_average_vars(model.variables)
