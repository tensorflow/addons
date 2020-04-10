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
"""Tests for Weighted Kappa loss."""

import numpy as np
from tensorflow_addons.losses.kappa_loss import WeightedKappaLoss


def weighted_kappa_loss_np(y_true, y_pred, weightage="quadratic", eps=1e-6):
    """Implemented in non-optimized python code to avoid mistakes"""

    num_samples, num_classes = y_true.shape
    numerator = 0
    true_classes = y_true.argmax(axis=1)
    for i in range(num_samples):
        true_class = true_classes[i]
        for j in range(num_classes):
            if weightage == "quadratic":
                w_ij = np.power(j - true_class, 2)
            else:
                w_ij = np.abs(j - true_class)
            numerator += w_ij * y_pred[i, j]

    classes_count = y_true.sum(axis=0)
    prob_sum = y_pred.sum(axis=0)

    denominator = 0
    for i in range(num_classes):
        inner_sum = 0
        for j in range(num_classes):
            if weightage == "quadratic":
                w_ij = np.power(i - j, 2)
            else:
                w_ij = np.abs(i - j)
            inner_sum += w_ij * prob_sum[j]
        denominator += inner_sum * classes_count[i] / num_samples
    return np.log(numerator / denominator + eps)


def test_linear_weighted_kappa_loss():
    y_true = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

    y_pred = np.array(
        [
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.5, 0.3, 0.1],
            [0.8, 0.05, 0.05, 0.1],
            [0.01, 0.09, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    kappa_loss = WeightedKappaLoss(num_classes=4, weightage="linear")
    loss = kappa_loss(y_true, y_pred)
    loss_np = weighted_kappa_loss_np(y_true, y_pred, weightage="linear")
    np.testing.assert_allclose(loss, loss_np, rtol=5, atol=5)


def test_quadratic_weighted_kappa_loss():
    y_true = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    y_pred = np.array(
        [
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.5, 0.3, 0.1],
            [0.8, 0.05, 0.05, 0.1],
            [0.01, 0.09, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    kappa_loss = WeightedKappaLoss(num_classes=4)
    loss = kappa_loss(y_true, y_pred)
    loss_np = weighted_kappa_loss_np(y_true, y_pred)
    np.testing.assert_allclose(loss, loss_np, rtol=5, atol=5)


def test_config():
    kappa_loss = WeightedKappaLoss(
        num_classes=4, weightage="linear", name="kappa_loss", epsilon=0.001,
    )
    assert kappa_loss.num_classes == 4
    assert kappa_loss.weightage == "linear"
    assert kappa_loss.name == "kappa_loss"
    np.testing.assert_allclose(kappa_loss.epsilon, 0.001, 1e-6)
