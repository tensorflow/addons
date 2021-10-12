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
"""Tests for SoftF1 loss."""


import pytest

import numpy as np
import tensorflow as tf
from tensorflow_addons.losses import softf1_loss, SoftF1Loss

def test_config():
    sl_obj = SoftF1Loss(reduction=tf.keras.losses.Reduction.NONE, name="softf1_loss")
    assert sl_obj.name == "softf1_loss"
    assert sl_obj.reduction == tf.keras.losses.Reduction.NONE

@pytest.mark.parametrize("from_logits", [True, False])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_soft_f1(dtype, from_logits):
    y_true = tf.constant([[1, 1, 1],
                        [1, 0, 0],
                        [1, 1, 0]], dtype=dtype)
    y_pred = tf.constant([[0.2, 0.6, 0.7],
                        [0.2, 0.6, 0.6],
                        [0.6, 0.8, 0.0]], dtype=dtype)
    expected_result = tf.constant([0.39710146], dtype=dtype)
    loss = softf1_loss(y_true, y_pred, from_logits=from_logits)
    np.testing.assert_allclose(loss, expected_result)

@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_keras_model(dtype):
    y_true = tf.constant([[1, 1, 1],
                        [1, 0, 0],
                        [1, 1, 0]], dtype=dtype)
    y_pred = tf.constant([[0.2, 0.6, 0.7],
                        [0.2, 0.6, 0.6],
                        [0.6, 0.8, 0.0]], dtype=dtype)
    expected_result = tf.constant([0.39710146], dtype=dtype)
    model = tf.keras.Sequential()
    model.compile(
        optimizer="adam",
        loss=SoftF1Loss()
    )
    loss = model.evaluate(y_true, y_pred, batch_size=3, steps=1)
    np.testing.assert_allclose(loss, expected_result)
