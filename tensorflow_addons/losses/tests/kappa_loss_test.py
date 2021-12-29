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
"""Tests for Weighted Kappa Loss."""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.losses.kappa_loss import WeightedKappaLoss


def weighted_kappa_loss_np(y_true, y_pred, weightage="quadratic", eps=1e-6):
    num_samples, num_classes = y_true.shape
    cat_labels = y_true.argmax(axis=1).reshape((-1, 1))
    label_mat = np.tile(cat_labels, (1, num_classes))
    row_label_vec = np.arange(num_classes).reshape((1, num_classes))
    label_mat_ = np.tile(row_label_vec, (num_samples, 1))
    if weightage == "linear":
        weight = np.abs(label_mat - label_mat_)
    else:
        weight = (label_mat - label_mat_) ** 2
    numerator = (y_pred * weight).sum()
    label_dist = y_true.sum(axis=0, keepdims=True)
    pred_dist = y_pred.sum(axis=0, keepdims=True)

    col_label_vec = row_label_vec.T
    row_mat = np.tile(row_label_vec, (num_classes, 1))
    col_mat = np.tile(col_label_vec, (1, num_classes))
    if weightage == "quadratic":
        weight_ = (col_mat - row_mat) ** 2
    else:
        weight_ = np.abs(col_mat - row_mat)
    weighted_pred_dist = np.matmul(weight_, pred_dist.T)
    denominator = np.matmul(label_dist, weighted_pred_dist).sum()
    denominator /= num_samples
    return np.log(np.nan_to_num(numerator / denominator) + eps)


def gen_labels_and_preds(num_samples, num_classes, seed):
    np.random.seed(seed)
    rands = np.random.uniform(size=(num_samples, num_classes))
    cat_labels = rands.argmax(axis=1)
    y_true = np.eye(num_classes, dtype="int")[cat_labels]
    y_pred = np.random.uniform(size=(num_samples, num_classes))
    y_pred /= y_pred.sum(axis=1, keepdims=True)
    return y_true, y_pred


@pytest.mark.parametrize("np_seed", [0, 1, 2, 3])
def test_linear_weighted_kappa_loss(np_seed):
    y_true, y_pred = gen_labels_and_preds(50, 4, np_seed)
    kappa_loss = WeightedKappaLoss(num_classes=4, weightage="linear")
    y_pred = y_pred.astype(np.float32)
    loss = kappa_loss(y_true, y_pred)
    loss_np = weighted_kappa_loss_np(y_true, y_pred, weightage="linear")
    np.testing.assert_allclose(loss, loss_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("np_seed", [0, 1, 2, 3])
def test_quadratic_weighted_kappa_loss(np_seed):
    y_true, y_pred = gen_labels_and_preds(100, 3, np_seed)
    kappa_loss = WeightedKappaLoss(num_classes=3)
    y_pred = y_pred.astype(np.float32)
    loss = kappa_loss(y_true, y_pred)
    loss_np = weighted_kappa_loss_np(y_true, y_pred)
    np.testing.assert_allclose(loss, loss_np, rtol=1e-5, atol=1e-5)


def test_config():
    kappa_loss = WeightedKappaLoss(
        num_classes=4, weightage="linear", name="kappa_loss", epsilon=0.001
    )
    assert kappa_loss.num_classes == 4
    assert kappa_loss.weightage == "linear"
    assert kappa_loss.name == "kappa_loss"
    np.testing.assert_allclose(kappa_loss.epsilon, 0.001, 1e-6)


def test_serialization():
    loss = WeightedKappaLoss(num_classes=3)
    tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))


def test_save_model(tmpdir):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((256, 256, 3)),
            tf.keras.layers.Dense(6, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss=WeightedKappaLoss(num_classes=6))
    model.save(str(tmpdir / "test.h5"))
