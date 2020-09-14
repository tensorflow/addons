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

from tensorflow_addons.optimizers.discriminative_layer_training import MultiOptimzer
from tensorflow_addons.utils import test_utils

def _dtypes_to_test(use_gpu):
    # Based on issue #347 in the following link,
    #        "https://github.com/tensorflow/addons/issues/347"
    # tf.half is not registered for 'ResourceScatterUpdate' OpKernel
    # for 'GPU' devices.
    # So we have to remove tf.half when testing with gpu.
    # The function "_DtypesToTest" is from
    #       "https://github.com/tensorflow/tensorflow/blob/5d4a6cee737a1dc6c20172a1dc1
    #        5df10def2df72/tensorflow/python/kernel_tests/conv_ops_3d_test.py#L53-L62"
    # TODO(WindQAQ): Clean up this in TF2.4

    if use_gpu:
        return [tf.float32, tf.float64]
    else:
        return [tf.half, tf.float32, tf.float64]


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32, tf.float64])
@pytest.mark.parametrize("serialize", [True, False])
def test_fit_layer_optimizer(dtype, device, serialize):
    # Test ensures that each optimizer is only optimizing its own layer with its learning rate

    if "gpu" in device and dtype == tf.float16:
        pytest.xfail("See https://github.com/tensorflow/addons/issues/347")

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
    optimizer = MultiOptimzer(opt_layer_pairs)

    model.compile(optimizer=optimizer, loss=loss)

    # serialize whole model including optimizer, clear the session, then reload the whole model.
    if serialize:
        model.save("test", save_format="tf")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model("test")

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


def test_serialization():

    model = tf.keras.Sequential(
        [tf.keras.Input(shape=[1]), tf.keras.layers.Dense(1), tf.keras.layers.Dense(1)]
    )

    opt1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    opt2 = tf.keras.optimizers.SGD(learning_rate=0)

    opt_layer_pairs = [(opt1, model.layers[0]), (opt2, model.layers[1])]

    optimizer = MultiOptimzer(opt_layer_pairs)
    config = tf.keras.optimizers.serialize(optimizer)

    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()