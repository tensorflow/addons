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
"""Matthews Correlation Coefficient Test."""


import tensorflow as tf

import numpy as np
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient


def test_config():
    # mcc object
    mcc1 = MatthewsCorrelationCoefficient(num_classes=1)
    assert mcc1.num_classes == 1
    assert mcc1.dtype == tf.float32
    # check configure
    mcc2 = MatthewsCorrelationCoefficient.from_config(mcc1.get_config())
    assert mcc2.num_classes == 1
    assert mcc2.dtype == tf.float32


def check_results(obj, value):
    np.testing.assert_allclose(value, obj.result().numpy(), atol=1e-6)


def test_binary_classes():
    gt_label = tf.constant([[1.0], [1.0], [1.0], [0.0]], dtype=tf.float32)
    preds = tf.constant([[1.0], [0.0], [1.0], [1.0]], dtype=tf.float32)
    # Initialize
    mcc = MatthewsCorrelationCoefficient(1)
    # Update
    mcc.update_state(gt_label, preds)
    # Check results
    check_results(mcc, [-0.33333334])


def test_multiple_classes():
    gt_label = tf.constant(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=tf.float32,
    )
    preds = tf.constant(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=tf.float32,
    )
    # Initialize
    mcc = MatthewsCorrelationCoefficient(3)
    mcc.update_state(gt_label, preds)
    # Check results
    check_results(mcc, [-0.33333334, 1.0, 0.57735026])


# Keras model API check
def test_keras_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="softmax"))
    mcc = MatthewsCorrelationCoefficient(num_classes=1)
    model.compile(
        optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy", mcc]
    )
    # data preparation
    data = np.random.random((10, 1))
    labels = np.random.random((10, 1))
    labels = np.where(labels > 0.5, 1.0, 0.0)
    model.fit(data, labels, epochs=1, batch_size=32, verbose=0)
