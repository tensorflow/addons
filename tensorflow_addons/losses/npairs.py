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
"""Implements npairs loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
@tf.function
def npairs_loss(y_true, y_pred):
    """Computes the npairs loss between `y_true` and `y_pred`.

    Npairs loss expects paired data where a pair is composed of samples from
    the same labels and each pairs in the minibatch have different labels.
    The loss takes each row of the pair-wise similarity matrix, `y_pred`,
    as logits and the remapped multi-class labels, `y_true`, as labels.

    The similarity matrix `y_pred` between two embedding matrices `a` and `b`
    with shape `[batch_size, hidden_size]` can be computed as follows:

    ```python
    # y_pred = a * b^T
    y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
    ```

    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multi-class labels.
      y_pred: 2-D float `Tensor` with shape `[batch_size, batch_size]` of
        similarity matrix between embedding matrices.

    Returns:
      npairs_loss: float scalar.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Expand to [batch_size, 1]
    y_true = tf.expand_dims(y_true, -1)
    y_true = tf.cast(tf.equal(y_true, tf.transpose(y_true)), y_pred.dtype)
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=y_pred, labels=y_true)

    return tf.math.reduce_mean(loss)


@keras_utils.register_keras_custom_object
class NpairsLoss(tf.keras.losses.Loss):
    """Computes the npairs loss between `y_true` and `y_pred`.

    Npairs loss expects paired data where a pair is composed of samples from
    the same labels and each pairs in the minibatch have different labels.
    The loss takes each row of the pair-wise similarity matrix, `y_pred`,
    as logits and the remapped multi-class labels, `y_true`, as labels.

    The similarity matrix `y_pred` between two embedding matrices `a` and `b`
    with shape `[batch_size, hidden_size]` can be computed as follows:

    ```python
    # y_pred = a * b^T
    y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
    ```

    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
      name: (Optional) name for the loss.
    """

    def __init__(self, name="npairs_loss"):
        super(NpairsLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE, name=name)

    def call(self, y_true, y_pred):
        return npairs_loss(y_true, y_pred)
