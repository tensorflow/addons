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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils.types import Number
from typeguard import typechecked
from typing import Optional


def _weighted_kappa_loss(
    y_true,
    y_pred,
    row_label_vec,
    col_label_vec,
    weight_mat,
    epsilon=1e-6,
    weightage="quadratic",
    dtype=tf.float32,
):
    y_true = tf.cast(y_true, dtype=dtype)
    labels = tf.matmul(y_true, col_label_vec)
    if weightage == "linear":
        weight = tf.abs(
            tf.tile(labels, [1, tf.shape(y_true)[1]])
            - tf.tile(row_label_vec, [tf.shape(y_true)[0], 1])
        )
        weight /= tf.cast(tf.shape(y_true)[1] - 1, dtype=dtype)
    else:
        weight = tf.pow(
            tf.tile(labels, [1, tf.shape(y_true)[1]])
            - tf.tile(row_label_vec, [tf.shape(y_true)[0], 1]),
            2,
        )
        weight /= tf.cast(tf.pow(tf.shape(y_true)[1] - 1, 2), dtype=dtype)
    numerator = tf.reduce_sum(weight * y_pred)

    denominator = tf.reduce_sum(
        tf.matmul(
            tf.reduce_sum(y_true, axis=0, keepdims=True),
            tf.matmul(
                weight_mat,
                tf.reduce_sum(y_pred, axis=0, keepdims=True),
                transpose_b=True,
            ),
        )
    )
    denominator /= tf.cast(tf.shape(y_true)[0], dtype=dtype)
    return tf.math.log(tf.math.divide_no_nan(numerator, denominator) + epsilon)


@tf.keras.utils.register_keras_serializable(package="Addons")
class WeightedKappaLoss(tf.keras.losses.Loss):
    """Implements the Weighted Kappa loss function.

    Weighted Kappa loss was introduced in the
    [Weighted kappa loss function for multi-class classification
    of ordinal data in deep learning]
    (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems.
    The loss value lies in [-inf, log 2], where log 2
     means the random prediction.

    Usage:

    ```python
    kappa_loss = WeightedKappaLoss(num_classes=4)
    y_true = tf.constant([[0, 0, 1, 0], [0, 1, 0, 0],
                          [1, 0, 0, 0], [0, 0, 0, 1]])
    y_pred = tf.constant([[0.1, 0.2, 0.6, 0.1], [0.1, 0.5, 0.3, 0.1],
                          [0.8, 0.05, 0.05, 0.1], [0.01, 0.09, 0.1, 0.8]])
    loss = kappa_loss(y_true, y_pred)
    print('Loss: ', loss.numpy())  # Loss: -1.1611923
    ```

    Usage with `tf.keras` API:
    ```python
    # outputs should be softmax results
    # if you want to weight the samples, just multiply the outputs
    # by the sample weight.
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tfa.losses.WeightedKappa(num_classes=4))
    ```
    """

    @typechecked
    def __init__(
        self,
        num_classes: int,
        weightage: Optional[str] = "quadratic",
        name: Optional[str] = "cohen_kappa_loss",
        epsilon: Optional[Number] = 1e-6,
        dtype: Optional[tf.DType] = tf.float32,
    ):
        """Creates a `WeightedKappa` instance.

        Args:
          num_classes: Number of unique classes in your dataset.
          weightage: (Optional) Weighting to be considered for calculating
            kappa statistics. A valid value is one of
            ['linear', 'quadratic']. Defaults to `quadratic` since it's
            mostly used.
          name: (Optional) String name of the metric instance.
          epsilon: (Optional) increment to avoid log zero,
            so the loss will be log(1 - k + epsilon), where k belongs to
            [-1, 1], usually you can use the default value which is 1e-6.
          dtype: (Optional) Data type of the metric result.
            Defaults to `tf.float32`.
        Raises:
          ValueError: If the value passed for `weightage` is invalid
            i.e. not any one of ['linear', 'quadratic']
        """

        super().__init__(
            name=name, reduction=tf.keras.losses.Reduction.NONE
        )

        if weightage not in ("linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")

        self.weightage = weightage
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.dtype = dtype
        label_vec = tf.range(num_classes, dtype=dtype)
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        if weightage == "linear":
            self.weight_mat = tf.abs(
                tf.tile(self.col_label_vec, [1, num_classes])
                - tf.tile(self.row_label_vec, [num_classes, 1]),
            ) / tf.cast(num_classes - 1, dtype=dtype)
        else:
            self.weight_mat = (
                tf.tile(self.col_label_vec, [1, num_classes])
                - tf.tile(self.row_label_vec, [num_classes, 1])
            ) ** 2 / tf.cast((num_classes - 1) ** 2, dtype=dtype)

    def call(self, y_true, y_pred):
        return _weighted_kappa_loss(
            y_true,
            y_pred,
            self.row_label_vec,
            self.col_label_vec,
            self.weight_mat,
            self.epsilon,
            self.weightage,
            self.dtype,
        )

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
            "epsilon": self.epsilon,
            "dtype": self.dtype,
        }
        base_config = super(WeightedKappaLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
