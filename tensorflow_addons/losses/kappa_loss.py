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
"""Implements Weighted kappa loss."""

import warnings
from typing import Optional

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils.types import Number


@tf.keras.utils.register_keras_serializable(package="Addons")
class WeightedKappaLoss(tf.keras.losses.Loss):
    r"""Implements the Weighted Kappa loss function.

    Weighted Kappa loss was introduced in the
    [Weighted kappa loss function for multi-class classification
    of ordinal data in deep learning]
    (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems.
    The loss value lies in $ [-\infty, \log 2] $, where $ \log 2 $
    means the random prediction.

    Usage:

    >>> kappa_loss = tfa.losses.WeightedKappaLoss(num_classes=4)
    >>> y_true = tf.constant([[0, 0, 1, 0], [0, 1, 0, 0],
    ...                  [1, 0, 0, 0], [0, 0, 0, 1]])
    >>> y_pred = tf.constant([[0.1, 0.2, 0.6, 0.1], [0.1, 0.5, 0.3, 0.1],
    ...                  [0.8, 0.05, 0.05, 0.1], [0.01, 0.09, 0.1, 0.8]])
    >>> loss = kappa_loss(y_true, y_pred)
    >>> loss
    <tf.Tensor: shape=(), dtype=float32, numpy=-1.1611925>

    Usage with `tf.keras` API:

    >>> model = tf.keras.Model()
    >>> model.compile('sgd', loss=tfa.losses.WeightedKappaLoss(num_classes=4))

    <... outputs should be softmax results
    if you want to weight the samples, just multiply the outputs
    by the sample weight ...>

    """

    @typechecked
    def __init__(
        self,
        num_classes: int,
        weightage: Optional[str] = "quadratic",
        name: Optional[str] = "cohen_kappa_loss",
        epsilon: Optional[Number] = 1e-6,
        dtype: Optional[tf.DType] = tf.float32,
        reduction: str = tf.keras.losses.Reduction.NONE,
    ):
        r"""Creates a `WeightedKappaLoss` instance.

        Args:
          num_classes: Number of unique classes in your dataset.
          weightage: (Optional) Weighting to be considered for calculating
            kappa statistics. A valid value is one of
            ['linear', 'quadratic']. Defaults to 'quadratic'.
          name: (Optional) String name of the metric instance.
          epsilon: (Optional) increment to avoid log zero,
            so the loss will be $ \log(1 - k + \epsilon) $, where $ k $ lies
            in $ [-1, 1] $. Defaults to 1e-6.
          dtype: (Optional) Data type of the metric result.
            Defaults to `tf.float32`.
        Raises:
          ValueError: If the value passed for `weightage` is invalid
            i.e. not any one of ['linear', 'quadratic']
        """

        super().__init__(name=name, reduction=reduction)

        warnings.warn(
            "The data type for `WeightedKappaLoss` defaults to "
            "`tf.keras.backend.floatx()`."
            "The argument `dtype` will be removed in Addons `0.12`.",
            DeprecationWarning,
        )

        if weightage not in ("linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")

        self.weightage = weightage
        self.num_classes = num_classes
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        label_vec = tf.range(num_classes, dtype=tf.keras.backend.floatx())
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        col_mat = tf.tile(self.col_label_vec, [1, num_classes])
        row_mat = tf.tile(self.row_label_vec, [num_classes, 1])
        if weightage == "linear":
            self.weight_mat = tf.abs(col_mat - row_mat)
        else:
            self.weight_mat = (col_mat - row_mat) ** 2

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=self.col_label_vec.dtype)
        y_pred = tf.cast(y_pred, dtype=self.weight_mat.dtype)
        batch_size = tf.shape(y_true)[0]
        cat_labels = tf.matmul(y_true, self.col_label_vec)
        cat_label_mat = tf.tile(cat_labels, [1, self.num_classes])
        row_label_mat = tf.tile(self.row_label_vec, [batch_size, 1])
        if self.weightage == "linear":
            weight = tf.abs(cat_label_mat - row_label_mat)
        else:
            weight = (cat_label_mat - row_label_mat) ** 2
        numerator = tf.reduce_sum(weight * y_pred)
        label_dist = tf.reduce_sum(y_true, axis=0, keepdims=True)
        pred_dist = tf.reduce_sum(y_pred, axis=0, keepdims=True)
        w_pred_dist = tf.matmul(self.weight_mat, pred_dist, transpose_b=True)
        denominator = tf.reduce_sum(tf.matmul(label_dist, w_pred_dist))
        denominator /= tf.cast(batch_size, dtype=denominator.dtype)
        loss = tf.math.divide_no_nan(numerator, denominator)
        return tf.math.log(loss + self.epsilon)

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}
