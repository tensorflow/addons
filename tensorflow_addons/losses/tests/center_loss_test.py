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
"""Tests for center loss."""


import numpy as np
import tensorflow as tf
from tensorflow_addons.losses import CenterLoss


def test_config():
    center_obj = CenterLoss(name="center_loss")
    np.testing.assert_equal(center_obj.name, "center_loss")
    # self.assertEqual(center_obj.reduction, tf.keras.losses.Reduction.SUM)


def test_zero_loss():
    center_obj = CenterLoss()
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=tf.dtypes.float32)
    loss = center_obj(y_true, y_pred)
    np.testing.assert_allclose(loss, 9.0)


def test_keras_model_compile():
    center_obj = CenterLoss()
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(400), tf.keras.layers.Dense(10),]
    )
    model.compile(loss=center_obj, optimizer="adam")
