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

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import TensorLike, Number


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def contrastive_loss(
    y_true: TensorLike, y_pred: TensorLike, margin: Number = 1.0
) -> tf.Tensor:
    r"""Computes the contrastive loss between `y_true` and `y_pred`.

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape `[batch_size, hidden_size]` can be computed
    as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],
    ...                 [1, 8]], dtype=tf.float16)
    >>> y_pred = tf.linalg.norm(a - b, axis=1)
    >>> y_pred
    <tf.Tensor: shape=(3,), dtype=float16, numpy=array([8.06 , 2.   , 4.473],
    dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        binary labels indicating positive vs negative pair.
      y_pred: 1-D float `Tensor` with shape `[batch_size]` of
        distances between two embedding matrices.
      margin: margin term in the loss definition.

    Returns:
      contrastive_loss: 1-D float `Tensor` with shape `[batch_size]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0)
    )


@tf.keras.utils.register_keras_serializable(package="Addons")
class ContrastiveLoss(LossFunctionWrapper):
    r"""Computes the contrastive loss between `y_true` and `y_pred`.

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    We expect labels `y_true` to be provided as 1-D integer `Tensor`
    with shape `[batch_size]` of binary integer labels. And `y_pred` must be
    1-D float `Tensor` with shape `[batch_size]` of distances between two
    embedding matrices.

    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape `[batch_size, hidden_size]` can be computed
    as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],[5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],[1, 8]], dtype=tf.float16)
    >>> y_pred = tf.linalg.norm(a - b, axis=1)
    >>> y_pred
    <tf.Tensor: shape=(3,), dtype=float16, numpy=array([8.06 , 2.   , 4.473],
    dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    Args:
      margin: `Float`, margin term in the loss definition.
        Default value is 1.0.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply.
        Default value is `SUM_OVER_BATCH_SIZE`.
      name: (Optional) name for the loss.
    """

    @typechecked
    def __init__(
        self,
        margin: Number = 1.0,
        reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "contrastive_loss",
    ):
        super().__init__(
            contrastive_loss, reduction=reduction, name=name, margin=margin
        )
