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
"""Tests for Discriminative Layer Training Optimizer for TensorFlow"""

import pytest
import numpy as np

from tensorflow_addons.optimizers.discriminative_layer_training import MultiOpt
import tensorflow as tf
from tensorflow_addons.utils import test_utils

# python -m flake8 tensorflow_addons/optimizers/discriminative_layer_training.py
# python -m black tensorflow_addons/optimizers/discriminative_layer_training.py


def _dtypes_to_test(use_gpu):
    # Based on issue #347 in the following link,
    #        "https://github.com/tensorflow/addons/issues/347"
    # tf.half is not registered for 'ResourceScatterUpdate' OpKernel
    # for 'GPU' devices.
    # So we have to remove tf.half when testing with gpu.
    # The function "_DtypesToTest" is from
    #       "https://github.com/tensorflow/tensorflow/blob/5d4a6cee737a1dc6c20172a1dc1
    #        5df10def2df72/tensorflow/python/kernel_tests/conv_ops_3d_test.py#L53-L62"
    if use_gpu:
        return [tf.float32, tf.float64]
    else:
        return [tf.half, tf.float32, tf.float64]

def _dtypes_with_checking_system(use_gpu, system):
    # Based on issue #36764 in the following link,
    #        "https://github.com/tensorflow/tensorflow/issues/36764"
    # tf.half is not registered for tf.linalg.svd function on Windows
    # CPU version.
    # So we have to remove tf.half when testing with Windows CPU version.
    if system == "Windows":
        return [tf.float32, tf.float64]
    else:
        return _dtypes_to_test(use_gpu)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32, tf.float64])
def test_fit_layer_optimizer(dtype, device):
    # Test ensures that each optimizer is only optimizing its own layer with its learning rate

    if "gpu" in device and dtype == tf.float16:
        pytest.xfail("See https://github.com/tensorflow/addons/issues/347")

    model = tf.keras.Sequential([tf.keras.Input(shape = [1]),
                               tf.keras.layers.Dense(1),
                                tf.keras.layers.Dense(1)
                ])

    x = np.array(np.ones([100]))
    y = np.array(np.ones([100]))

    weights_before_train = (model.layers[0].weights[0].numpy(),
                            model.layers[1].weights[0].numpy())

    opt1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    opt2 = tf.keras.optimizers.SGD(learning_rate=0)

    opt_layer_pairs = [(opt1, model.layers[0]), (opt2, model.layers[1])]

    loss = tf.keras.losses.MSE
    optimizer = MultiOpt(opt_layer_pairs)

    model.compile(optimizer=optimizer,
            loss = loss)

    model.fit(x, y, batch_size=8, epochs=10)

    weights_after_train = (model.layers[0].weights[0].numpy(),
                            model.layers[1].weights[0].numpy())

    with np.testing.assert_raises(AssertionError):
        # expect weights to be different for layer 1
        test_utils.assert_allclose_according_to_type(weights_before_train[0], weights_after_train[0])

    # expect weights to be same for layer 2
    test_utils.assert_allclose_according_to_type(weights_before_train[1], weights_after_train[1])
