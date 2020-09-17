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
"""Tests Hamming metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from scipy import stats

import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers

from tensorflow_addons.metrics import KendallsTau, kendalls_tau


def test_config():
    kl_obj = KendallsTau()
    assert kl_obj.name == "kendalls_tau"
    assert kl_obj.dtype == tf.float32


def test_kendall_tau():
    x1 = [12, 2, 1, 12, 2]
    x2 = [1, 4, 7, 1, 0]
    expected = stats.kendalltau(x1, x2)[0]
    res = kendalls_tau(tf.constant(x1, tf.float32), tf.constant(x2, tf.float32))
    np.testing.assert_allclose(expected, res.numpy(), atol=1e-5)


def test_kendall_tau_float():
    x1 = [0.12, 0.02, 0.01, 0.12, 0.02]
    x2 = [0.1, 0.4, 0.7, 0.1, 0.0]
    expected = stats.kendalltau(x1, x2)[0]
    res = kendalls_tau(tf.constant(x1, tf.float32), tf.constant(x2, tf.float32))
    np.testing.assert_allclose(expected, res.numpy(), atol=1e-5)


def test_kendall_random_lists():
    left = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9]
    for _ in range(10):
        right = random.sample(left, len(left))
        expected = stats.kendalltau(left, right)[0]
        res = kendalls_tau(
            tf.constant(left, tf.float32), tf.constant(right, tf.float32)
        )
        np.testing.assert_allclose(expected, res.numpy(), atol=1e-5)


def test_keras_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(1,)))
    model.add(layers.Dense(1, kernel_initializer="ones"))
    kt = KendallsTau()
    model.compile(optimizer="rmsprop", loss="mae", metrics=[kt])
    data = np.array([[0.12], [0.02], [0.01], [0.12], [0.02]])
    labels = np.array([0.1, 0.4, 0.7, 0.1, 0.0])
    history = model.fit(data, labels, epochs=1, batch_size=5, verbose=0)
    expected = stats.kendalltau(np.array(data).flat, labels)[0]
    np.testing.assert_allclose(expected, history.history["kendalls_tau"], atol=1e-5)


def test_averaging_tau_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(1,)))
    model.add(layers.Dense(1, kernel_initializer="ones"))
    kt = KendallsTau()
    model.compile(optimizer="rmsprop", loss="mae", metrics=[kt])
    data = np.array([[5], [3], [2], [1], [4], [1], [2], [3], [4], [5]])
    labels = np.array([1, 2, 2, 3, 6, 10, 11, 12, 13, 14])
    history = model.fit(data, labels, epochs=1, batch_size=5, verbose=0, shuffle=False)
    expected = np.mean(
        [
            stats.kendalltau(data[0:5].flat, labels[0:5])[0],
            stats.kendalltau(data[5:].flat, labels[5:])[0],
        ]
    )
    np.testing.assert_allclose(expected, history.history["kendalls_tau"], atol=1e-5)
