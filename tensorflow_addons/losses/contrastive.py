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
"""Implements contrastive loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
@tf.function
def contrastive_loss(y_true, y_pred, margin=1.0):
    """Computes the contrastive loss between `y_true` and `y_pred`.

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape [batch_size, hidden_size] can be computed
    as follows:

    ```python
    # y_pred = \sqrt (\sum_i (a[:, i] - b[:, i])^2)
    y_pred = tf.linalg.norm(a - b, axis=1)
    ```

    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        binary labels indicating positive vs negative pair.
      y_pred: 1-D float `Tensor` with shape [batch_size] of
        distances between two embedding matrices.
      margin: margin term in the loss definition.

    Returns:
      contrastive_loss: 1-D float `Tensor` with shape [batch_size].
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return (
        y_true * tf.math.square(y_pred) +
        (1. - y_true) * tf.math.square(tf.math.maximum(margin - y_pred, 0.)))


@keras_utils.register_keras_custom_object
class ContrastiveLoss(tf.keras.losses.Loss):
    """Computes the contrastive loss between `y_true` and `y_pred`.

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    We expect labels `y_true` to be provided as 1-D integer `Tensor`
    with shape [batch_size] of binary integer labels. And `y_pred` must be
    1-D float `Tensor` with shape [batch_size] of distances between two
    embedding matrices.

    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape [batch_size, hidden_size] can be computed
    as follows:

    ```python
    # y_pred = \sqrt (\sum_i (a[:, i] - b[:, i])^2)
    y_pred = tf.linalg.norm(a - b, axis=1)
    ```

    Args:
      margin: `Float`, margin term in the loss definition.
        Default value is 1.0.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply.
        Default value is `SUM_OVER_BATCH_SIZE`.
      name: (Optional) name for the loss.
    """

    def __init__(self,
                 margin=1.0,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="contrasitve_loss"):
        super(ContrastiveLoss, self).__init__(reduction=reduction, name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        return contrastive_loss(y_true, y_pred, self.margin)

    def get_config(self):
        config = {
            "margin": self.margin,
        }
        base_config = super(ContrastiveLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
