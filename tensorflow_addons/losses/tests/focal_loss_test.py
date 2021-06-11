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


import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.losses import focal_loss


def test_config():
    bce_obj = focal_loss.SigmoidFocalCrossEntropy(
        reduction=tf.keras.losses.Reduction.NONE, name="sigmoid_focal_crossentropy"
    )
    assert bce_obj.name == "sigmoid_focal_crossentropy"
    assert bce_obj.reduction == tf.keras.losses.Reduction.NONE


def test_keras_model_compile():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )
    model.compile(loss="Addons>sigmoid_focal_crossentropy")


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("y_pred_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("y_true_dtype", [np.float32, np.uint8])
@pytest.mark.parametrize("from_logits", [True, False])
def test_without_logits(y_pred_dtype, y_true_dtype, from_logits):
    y_pred = np.asarray([[0.97], [0.91], [0.73], [0.27], [0.09], [0.03]]).astype(
        y_pred_dtype
    )
    y_true = np.asarray([[1], [1], [1], [0], [0], [0]]).astype(y_true_dtype)
    if from_logits:
        y_pred = np.log(y_pred / (1.0 - y_pred))

    # When alpha and gamma are None, the result is equal to BCE.
    fl = focal_loss.sigmoid_focal_crossentropy(
        y_true=y_true, y_pred=y_pred, alpha=None, gamma=None, from_logits=from_logits
    ).numpy()
    bce = tf.keras.losses.binary_crossentropy(
        y_true, y_pred, from_logits=from_logits
    ).numpy()
    np.testing.assert_allclose(fl, bce)

    # When gamma is 2.0.
    fl = focal_loss.sigmoid_focal_crossentropy(
        y_true=y_true, y_pred=y_pred, alpha=None, gamma=2.0, from_logits=from_logits
    ).numpy()
    order_of_ratio = np.power(10.0, np.floor(np.log10(bce / fl)))
    pow_values = np.asarray([1000, 100, 10, 10, 100, 1000])
    np.testing.assert_allclose(order_of_ratio, pow_values)
