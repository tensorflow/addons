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
"""Tests for focal loss."""

import sys

import pytest
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_addons.losses import (
    sigmoid_focal_crossentropy,
    SigmoidFocalCrossEntropy,
)


def test_config():
    bce_obj = SigmoidFocalCrossEntropy(
        reduction=tf.keras.losses.Reduction.NONE, name="sigmoid_focal_crossentropy"
    )
    assert bce_obj.name == "sigmoid_focal_crossentropy"
    assert bce_obj.reduction == tf.keras.losses.Reduction.NONE


def to_logit(prob):
    logit = np.log(prob / (1.0 - prob))
    return logit


# Test with logits
def test_with_logits():
    # predictiions represented as logits
    prediction_tensor = tf.constant(
        [
            [to_logit(0.97)],
            [to_logit(0.91)],
            [to_logit(0.73)],
            [to_logit(0.27)],
            [to_logit(0.09)],
            [to_logit(0.03)],
        ],
        tf.float32,
    )
    # Ground truth
    target_tensor = tf.constant([[1], [1], [1], [0], [0], [0]], tf.float32)

    fl = sigmoid_focal_crossentropy(
        y_true=target_tensor,
        y_pred=prediction_tensor,
        from_logits=True,
        alpha=None,
        gamma=None,
    )
    bce = tf.reduce_sum(
        K.binary_crossentropy(target_tensor, prediction_tensor, from_logits=True),
        axis=-1,
    )

    # When alpha and gamma are None, it should be equal to BCE
    np.testing.assert_allclose(fl, bce)

    # When gamma==2.0
    fl = sigmoid_focal_crossentropy(
        y_true=target_tensor,
        y_pred=prediction_tensor,
        from_logits=True,
        alpha=None,
        gamma=2.0,
    )

    # order_of_ratio = np.power(10, np.floor(np.log10(bce/FL)))
    order_of_ratio = tf.pow(10.0, tf.math.floor(log10(bce / fl)))
    pow_values = tf.constant([1000, 100, 10, 10, 100, 1000])
    np.testing.assert_allclose(order_of_ratio, pow_values)


def test_keras_model_compile():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )
    model.compile(loss="Addons>sigmoid_focal_crossentropy")


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


# Test without logits
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_without_logits():
    # predictiions represented as logits
    prediction_tensor = tf.constant(
        [[0.97], [0.91], [0.73], [0.27], [0.09], [0.03]], tf.float32
    )
    # Ground truth
    target_tensor = tf.constant([[1], [1], [1], [0], [0], [0]], tf.float32)

    fl = sigmoid_focal_crossentropy(
        y_true=target_tensor, y_pred=prediction_tensor, alpha=None, gamma=None
    )
    bce = tf.reduce_sum(
        K.binary_crossentropy(target_tensor, prediction_tensor), axis=-1
    )

    # When alpha and gamma are None, it should be equal to BCE
    assert np.allclose(fl, bce)

    # When gamma==2.0
    fl = sigmoid_focal_crossentropy(
        y_true=target_tensor, y_pred=prediction_tensor, alpha=None, gamma=2.0
    )

    order_of_ratio = tf.pow(10.0, tf.math.floor(log10(bce / fl)))
    pow_values = tf.constant([1000, 100, 10, 10, 100, 1000])
    assert np.allclose(order_of_ratio, pow_values)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
