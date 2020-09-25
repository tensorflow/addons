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

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.optimizers.discriminative_layer_training import MultiOptimizer
from tensorflow_addons.utils import test_utils


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("serialize", [True, False])
def test_fit_layer_optimizer(device, serialize, tmpdir):
    model = tf.keras.Sequential(
        [tf.keras.Input(shape=[1]), tf.keras.layers.Dense(1), tf.keras.layers.Dense(1)]
    )

    x = np.array(np.ones([100]))
    y = np.array(np.ones([100]))

    weights_before_train = (
        model.layers[0].weights[0].numpy(),
        model.layers[1].weights[0].numpy(),
    )

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
        model = tf.keras.models.load_model(tmpdir)

    model.fit(x, y, batch_size=8, epochs=10)

    weights_after_train = (
        model.layers[0].weights[0].numpy(),
        model.layers[1].weights[0].numpy(),
    )

    with np.testing.assert_raises(AssertionError):
        # expect weights to be different for layer 1
        test_utils.assert_allclose_according_to_type(
            weights_before_train[0], weights_after_train[0]
        )

    # expect weights to be same for layer 2
    test_utils.assert_allclose_according_to_type(
        weights_before_train[1], weights_after_train[1]
    )


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

    x = np.random.rand(128, 4)
    y = np.random.rand(128, 32)
    model.fit(x, y, batch_size=32, epochs=10)

    loss = model.evaluate(x, y)
    assert loss < 0.15

    weights_after_train = [
        [weight.numpy() for weight in layer.weights] for layer in model.layers
    ]

    for w_before, w_after in zip(weights_before_train[0], weights_after_train[0]):
        test_utils.assert_allclose_according_to_type(w_before, w_after)

    for layer_before, layer_after in zip(
        weights_before_train[1:], weights_after_train[1:]
    ):
        for w_before, w_after in zip(layer_before, layer_after):
            with np.testing.assert_raises(AssertionError):
                test_utils.assert_allclose_according_to_type(w_before, w_after)


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
