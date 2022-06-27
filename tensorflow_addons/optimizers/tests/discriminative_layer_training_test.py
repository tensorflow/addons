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
"""Tests for Discriminative Layer Training Optimizer for TensorFlow."""

from math import ceil

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.optimizers.discriminative_layer_training import MultiOptimizer
from tensorflow_addons.utils import test_utils


def assert_list_allclose(a, b):
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)


def assert_list_not_allclose(a, b):
    for x, y in zip(a, b):
        test_utils.assert_not_allclose(x, y)


def assert_sync_iterations(opt, desired):
    sub_opts = [specs["optimizer"] for specs in opt.optimizer_specs]
    for o in [opt] + sub_opts:
        np.testing.assert_equal(o.iterations.numpy(), desired)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("serialize", [True, False])
def test_fit_layer_optimizer(device, serialize, tmpdir):
    model = tf.keras.Sequential(
        [tf.keras.Input(shape=[1]), tf.keras.layers.Dense(1), tf.keras.layers.Dense(1)]
    )

    x = np.array(np.ones([100]))
    y = np.array(np.ones([100]))

    dense1_weights_before_train = [weight.numpy() for weight in model.layers[0].weights]
    dense2_weights_before_train = [weight.numpy() for weight in model.layers[1].weights]

    opt1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    opt2 = tf.keras.optimizers.SGD(learning_rate=0)

    opt_layer_pairs = [(opt1, model.layers[0]), (opt2, model.layers[1])]

    loss = tf.keras.losses.MSE
    optimizer = MultiOptimizer(opt_layer_pairs)

    model.compile(optimizer=optimizer, loss=loss)

    # serialize whole model including optimizer, clear the session, then reload the whole model.
    if serialize:
        model.save(str(tmpdir), save_format="tf")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(str(tmpdir))

    model.fit(x, y, batch_size=8, epochs=10)

    dense1_weights_after_train = [weight.numpy() for weight in model.layers[0].weights]
    dense2_weights_after_train = [weight.numpy() for weight in model.layers[1].weights]

    assert_list_not_allclose(dense1_weights_before_train, dense1_weights_after_train)
    assert_list_allclose(dense2_weights_before_train, dense2_weights_after_train)
    assert_sync_iterations(model.optimizer, desired=ceil(100 / 8) * 10)


def test_list_of_layers():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(32),
        ]
    )

    optimizers_and_layers = [
        (tf.keras.optimizers.SGD(learning_rate=0.0), model.layers[0]),
        (tf.keras.optimizers.Adam(), model.layers[1]),
        (tf.keras.optimizers.Adam(), model.layers[2:]),
    ]

    weights_before_train = [
        [weight.numpy() for weight in layer.weights] for layer in model.layers
    ]

    multi_optimizer = MultiOptimizer(optimizers_and_layers)
    model.compile(multi_optimizer, loss="mse")

    x = np.ones((128, 4)).astype(np.float32)
    y = np.ones((128, 32)).astype(np.float32)
    model.fit(x, y, batch_size=32, epochs=10)

    weights_after_train = [
        [weight.numpy() for weight in layer.weights] for layer in model.layers
    ]

    assert_list_allclose(weights_before_train[0], weights_after_train[0])

    for layer_before, layer_after in zip(
        weights_before_train[1:], weights_after_train[1:]
    ):
        assert_list_not_allclose(layer_before, layer_after)

    assert_sync_iterations(model.optimizer, desired=ceil(128 / 32) * 10)


def test_model():
    inputs = tf.keras.Input(shape=(4,))
    output = tf.keras.layers.Dense(16)(inputs)
    output = tf.keras.layers.Dense(16)(output)
    output = tf.keras.layers.Dense(32)(output)
    output = tf.keras.layers.Dense(32)(output)
    model = tf.keras.Model(inputs, output)

    # Adam optimizer on the whole model and an additional SGD on the last layer.
    optimizers_and_layers = [
        (tf.keras.optimizers.Adam(), model),
        (tf.keras.optimizers.SGD(), model.layers[-1]),
    ]

    multi_optimizer = MultiOptimizer(optimizers_and_layers)
    model.compile(multi_optimizer, loss="mse")

    x = np.ones((128, 4)).astype(np.float32)
    y = np.ones((128, 32)).astype(np.float32)
    model.fit(x, y, batch_size=32, epochs=10)

    assert_sync_iterations(model.optimizer, desired=ceil(128 / 32) * 10)


def test_subclass_model():
    class Block(tf.keras.Model):
        def __init__(self, units):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(units)
            self.dense2 = tf.keras.layers.Dense(units)

        def call(self, x):
            return self.dense2(self.dense1(x))

    class Custom(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.block1 = Block(16)
            self.block2 = Block(32)

        def call(self, x):
            return self.block2(self.block1(x))

    model = Custom()
    model.build(input_shape=(None, 4))

    optimizers_and_layers = [
        (tf.keras.optimizers.SGD(learning_rate=0.0), model.block1),
        (tf.keras.optimizers.Adam(), model.block2),
    ]

    block1_weights_before_train = [weight.numpy() for weight in model.block1.weights]
    block2_weights_before_train = [weight.numpy() for weight in model.block2.weights]

    multi_optimizer = MultiOptimizer(optimizers_and_layers)

    x = np.ones((128, 4)).astype(np.float32)
    y = np.ones((128, 32)).astype(np.float32)
    mse = tf.keras.losses.MeanSquaredError()

    for _ in range(10):
        for i in range(0, 128, 32):
            x_batch = x[i : i + 32]
            y_batch = y[i : i + 32]
            with tf.GradientTape() as tape:
                loss = mse(y_batch, model(x_batch))

            grads = tape.gradient(loss, model.trainable_variables)
            multi_optimizer.apply_gradients(zip(grads, model.trainable_variables))

    block1_weights_after_train = [weight.numpy() for weight in model.block1.weights]
    block2_weights_after_train = [weight.numpy() for weight in model.block2.weights]

    assert_list_allclose(block1_weights_before_train, block1_weights_after_train)
    assert_list_not_allclose(block2_weights_before_train, block2_weights_after_train)
    assert_sync_iterations(multi_optimizer, desired=ceil(128 / 32) * 10)


def test_pretrained_model():
    resnet = tf.keras.applications.ResNet50(include_top=False, weights=None)
    dense = tf.keras.layers.Dense(32)
    model = tf.keras.Sequential([resnet, dense])

    resnet_weights_before_train = [
        weight.numpy() for weight in resnet.trainable_weights
    ]
    dense_weights_before_train = [weight.numpy() for weight in dense.weights]

    optimizers_and_layers = [(tf.keras.optimizers.SGD(), dense)]

    multi_optimizer = MultiOptimizer(optimizers_and_layers)
    model.compile(multi_optimizer, loss="mse")

    x = np.ones((128, 32, 32, 3)).astype(np.float32)
    y = np.ones((128, 32)).astype(np.float32)
    model.fit(x, y, batch_size=32)

    resnet_weights_after_train = [weight.numpy() for weight in resnet.trainable_weights]
    dense_weights_after_train = [weight.numpy() for weight in dense.weights]

    assert_list_allclose(resnet_weights_before_train, resnet_weights_after_train)
    assert_list_not_allclose(dense_weights_before_train, dense_weights_after_train)
    assert_sync_iterations(model.optimizer, desired=ceil(128 / 32))


def test_nested_model():
    def get_model():
        inputs = tf.keras.Input(shape=(4,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        return tf.keras.Model(inputs, outputs)

    model1 = get_model()
    model2 = get_model()
    model3 = get_model()

    inputs = tf.keras.Input(shape=(4,))
    y1 = model1(inputs)
    y2 = model2(inputs)
    y3 = model3(inputs)
    outputs = tf.keras.layers.Average()([y1, y2, y3])
    model = tf.keras.Model(inputs, outputs)

    optimizers_and_layers = [
        (tf.keras.optimizers.SGD(), model1),
        (tf.keras.optimizers.SGD(learning_rate=0.0), model2),
        (tf.keras.optimizers.SGD(), model3),
    ]

    model1_weights_before_train = [weight.numpy() for weight in model1.weights]
    model2_weights_before_train = [weight.numpy() for weight in model2.weights]
    model3_weights_before_train = [weight.numpy() for weight in model3.weights]

    multi_optimizer = MultiOptimizer(optimizers_and_layers)

    model.compile(multi_optimizer, loss="mse")

    x = np.ones((128, 4)).astype(np.float32)
    y = np.ones((128, 32)).astype(np.float32)
    model.fit(x, y)

    model1_weights_after_train = [weight.numpy() for weight in model1.weights]
    model2_weights_after_train = [weight.numpy() for weight in model2.weights]
    model3_weights_after_train = [weight.numpy() for weight in model3.weights]

    assert_list_not_allclose(model1_weights_before_train, model1_weights_after_train)
    assert_list_allclose(model2_weights_before_train, model2_weights_after_train)
    assert_list_not_allclose(model3_weights_before_train, model3_weights_after_train)
    assert_sync_iterations(model.optimizer, desired=ceil(128 / 32))


def test_serialization():
    model = tf.keras.Sequential(
        [tf.keras.Input(shape=[1]), tf.keras.layers.Dense(1), tf.keras.layers.Dense(1)]
    )

    opt1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    opt2 = tf.keras.optimizers.SGD(learning_rate=0)

    opt_layer_pairs = [(opt1, model.layers[0]), (opt2, model.layers[1])]

    optimizer = MultiOptimizer(opt_layer_pairs)
    config = tf.keras.optimizers.serialize(optimizer)

    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()


def test_serialization_after_training(tmpdir):
    x = np.array(np.ones([100]))
    y = np.array(np.ones([100]))
    model = tf.keras.Sequential(
        [tf.keras.Input(shape=[1]), tf.keras.layers.Dense(1), tf.keras.layers.Dense(1)]
    )

    opt1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    opt2 = tf.keras.optimizers.SGD(learning_rate=0)

    opt_layer_pairs = [(opt1, model.layers[0]), (opt2, model.layers[1])]

    optimizer = MultiOptimizer(opt_layer_pairs)

    # Train the model for a few epochs.
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    model.fit(x, y)

    # Verify the optimizer can still be serialized (saved).
    model.save(str(tmpdir))
    loaded_model = tf.keras.models.load_model(str(tmpdir))
    old_config = model.optimizer.get_config()
    new_config = loaded_model.optimizer.get_config()
    # Verify the loaded model has the same optimizer as before.
    assert len(old_config["optimizer_specs"]) == len(new_config["optimizer_specs"])
    for old_optimizer_spec, new_optimizer_spec in zip(
        old_config["optimizer_specs"], new_config["optimizer_specs"]
    ):
        assert old_optimizer_spec["weights"] == new_optimizer_spec["weights"]
        assert (
            old_optimizer_spec["optimizer"].get_config()
            == new_optimizer_spec["optimizer"].get_config()
        )
