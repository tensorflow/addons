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
"""Tests for contrastive loss."""

import numpy as np
import tensorflow as tf
from tensorflow_addons.losses import contrastive


def test_config():
    cl_obj = contrastive.ContrastiveLoss(
        reduction=tf.keras.losses.Reduction.SUM, name="cl"
    )
    assert cl_obj.name == "cl"
    assert cl_obj.reduction == tf.keras.losses.Reduction.SUM


def test_zero_loss():
    cl_obj = contrastive.ContrastiveLoss()
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=tf.dtypes.float32)
    loss = cl_obj(y_true, y_pred)
    np.testing.assert_allclose(loss, 0.0)


def test_unweighted():
    cl_obj = contrastive.ContrastiveLoss()
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5], dtype=tf.dtypes.float32)
    loss = cl_obj(y_true, y_pred)

    # Loss = y * (y`)^2 + (1 - y) * (max(m - y`, 0))^2
    #      = [max(1 - 0.1, 0)^2, max(1 - 0.3, 0)^2,
    #         1.3^2, 0.7^2, max(1 - 1.1, 0)^2, 0.5^2]
    #      = [0.9^2, 0.7^2, 1.3^2, 0.7^2, 0^2, 0.5^2]
    #      = [0.81, 0.49, 1.69, 0.49, 0, 0.25]
    # Reduced loss = (0.81 + 0.49 + 1.69 + 0.49 + 0 + 0.25) / 6
    #              = 0.621666

    np.testing.assert_allclose(loss, 0.621666, atol=1e-6, rtol=1e-6)


def test_sample_weighted():
    cl_obj = contrastive.ContrastiveLoss()
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5], dtype=tf.dtypes.float32)
    sample_weight = tf.constant([1.2, 0.8, 0.5, 0.4, 1.5, 1.0], dtype=tf.dtypes.float32)
    loss = cl_obj(y_true, y_pred, sample_weight=sample_weight)

    # Loss = y * (y`)^2 + (1 - y) * (max(m - y`, 0))^2
    #      = [max(1 - 0.1, 0)^2, max(1 - 0.3, 0)^2,
    #         1.3^2, 0.7^2, max(1 - 1.1, 0)^2, 0.5^2]
    #      = [0.9^2, 0.7^2, 1.3^2, 0.7^2, 0^2, 0.5^2]
    #      = [0.81, 0.49, 1.69, 0.49, 0, 0.25]
    # Weighted loss = [0.81 * 1.2, 0.49 * 0.8, 1.69 * 0.5,
    #                  0.49 * 0.4, 0 * 1.5, 0.25 * 1.0]
    #               = [0.972, 0.392, 0.845, 0.196, 0, 0.25]
    # Reduced loss = (0.972 + 0.392 + 0.845 + 0.196 + 0 + 0.25) / 6
    #              = 0.4425

    np.testing.assert_allclose(loss, 0.4425)


def test_non_default_margin():
    cl_obj = contrastive.ContrastiveLoss(margin=2.0)
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5], dtype=tf.dtypes.float32)
    loss = cl_obj(y_true, y_pred)

    # Loss = y * (y`)^2 + (1 - y) * (max(m - y`, 0))^2
    #      = [max(2 - 0.1, 0)^2, max(2 - 0.3, 0)^2,
    #         1.3^2, 0.7^2, max(2 - 1.1, 0)^2, 0.5^2]
    #      = [1.9^2, 1.7^2, 1.3^2, 0.7^2, 0.9^2, 0.5^2]
    #      = [3.61, 2.89, 1.69, 0.49, 0.81, 0.25]
    # Reduced loss = (3.61 + 2.89 + 1.69 + 0.49 + 0.81 + 0.25) / 6
    #              = 1.623333

    np.testing.assert_allclose(loss, 1.623333, atol=1e-6, rtol=1e-6)


def test_scalar_weighted():
    cl_obj = contrastive.ContrastiveLoss()
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5], dtype=tf.dtypes.float32)
    loss = cl_obj(y_true, y_pred, sample_weight=6.0)

    # Loss = y * (y`)^2 + (1 - y) * (max(m - y`, 0))^2
    #      = [max(1 - 0.1, 0)^2, max(1 - 0.3, 0)^2,
    #         1.3^2, 0.7^2, max(1 - 1.1, 0)^2, 0.5^2]
    #      = [0.9^2, 0.7^2, 1.3^2, 0.7^2, 0^2, 0.5^2]
    #      = [0.81, 0.49, 1.69, 0.49, 0, 0.25]
    # Weighted loss = [0.81 * 6, 0.49 * 6, 1.69 * 6,
    #                  0.49 * 6, 0 * 6, 0.25 * 6]
    # Reduced loss = (0.81 + 0.49 + 1.69 + 0.49 + 0 + 0.25) * 6 / 6
    #              = 3.73

    np.testing.assert_allclose(loss, 3.73, atol=1e-6, rtol=1e-6)


def test_zero_weighted():
    cl_obj = contrastive.ContrastiveLoss()
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5], dtype=tf.dtypes.float32)
    loss = cl_obj(y_true, y_pred, sample_weight=0.0)
    np.testing.assert_allclose(loss, 0.0)


def test_no_reduction():
    cl_obj = contrastive.ContrastiveLoss(reduction=tf.keras.losses.Reduction.NONE)
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5], dtype=tf.dtypes.float32)
    loss = cl_obj(y_true, y_pred)

    # Loss = y * (y`)^2 + (1 - y) * (max(m - y`, 0))^2
    #      = [max(1 - 0.1, 0)^2, max(1 - 0.3, 0)^2,
    #         1.3^2, 0.7^2, max(1 - 1.1, 0)^2, 0.5^2]
    #      = [0.9^2, 0.7^2, 1.3^2, 0.7^2, 0^2, 0.5^2]
    #      = [0.81, 0.49, 1.69, 0.49, 0, 0.25]

    np.testing.assert_allclose(loss, [0.81, 0.49, 1.69, 0.49, 0.0, 0.25], rtol=1e-5)


def test_sum_reduction():
    cl_obj = contrastive.ContrastiveLoss(reduction=tf.keras.losses.Reduction.SUM)
    y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
    y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5], dtype=tf.dtypes.float32)
    loss = cl_obj(y_true, y_pred)

    # Loss = y * (y`)^2 + (1 - y) * (max(m - y`, 0))^2
    #      = [max(1 - 0.1, 0)^2, max(1 - 0.3, 0)^2,
    #         1.3^2, 0.7^2, max(1 - 1.1, 0)^2, 0.5^2]
    #      = [0.9^2, 0.7^2, 1.3^2, 0.7^2, 0^2, 0.5^2]
    #      = [0.81, 0.49, 1.69, 0.49, 0, 0.25]
    # Reduced loss = 0.81 + 0.49 + 1.69 + 0.49 + 0 + 0.25
    #              = 3.73

    np.testing.assert_allclose(loss, 3.73)
