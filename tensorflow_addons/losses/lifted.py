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
"""Implements lifted_struct_loss."""

import tensorflow as tf
from tensorflow_addons.losses import metric_learning

from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from typeguard import typechecked
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def lifted_struct_loss(
    labels: TensorLike, embeddings: TensorLike, margin: FloatTensorLike = 1.0
) -> tf.Tensor:
    """Computes the lifted structured loss.

    Args:
      labels: 1-D tf.int32 `Tensor` with shape `[batch_size]` of
        multiclass integer labels.
      embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
        not be l2 normalized.
      margin: Float, margin term in the loss definition.

    Returns:
      lifted_loss: float scalar with dtype of embeddings.
    """
    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pairwise_distances = metric_learning.pairwise_distance(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    diff = margin - pairwise_distances
    mask = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    # Safe maximum: Temporarily shift negative distances
    #   above zero before taking max.
    #     this is to take the max only among negatives.
    row_minimums = tf.math.reduce_min(diff, 1, keepdims=True)
    row_negative_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(diff - row_minimums, mask), 1, keepdims=True
        )
        + row_minimums
    )

    # Compute the loss.
    # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
    #   where m_i is the max of alpha - negative D_i's.
    # This matches the Caffe loss layer implementation at:
    #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp

    max_elements = tf.math.maximum(
        row_negative_maximums, tf.transpose(row_negative_maximums)
    )
    diff_tiled = tf.tile(diff, [batch_size, 1])
    mask_tiled = tf.tile(mask, [batch_size, 1])
    max_elements_vect = tf.reshape(tf.transpose(max_elements), [-1, 1])

    loss_exp_left = tf.reshape(
        tf.math.reduce_sum(
            tf.math.multiply(tf.math.exp(diff_tiled - max_elements_vect), mask_tiled),
            1,
            keepdims=True,
        ),
        [batch_size, batch_size],
    )

    loss_mat = max_elements + tf.math.log(loss_exp_left + tf.transpose(loss_exp_left))
    # Add the positive distance.
    loss_mat += pairwise_distances

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
    num_positives = tf.math.reduce_sum(mask_positives) / 2.0

    lifted_loss = tf.math.truediv(
        0.25
        * tf.math.reduce_sum(
            tf.math.square(
                tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
            )
        ),
        num_positives,
    )

    if convert_to_float32:
        return tf.cast(lifted_loss, embeddings.dtype)
    else:
        return lifted_loss


@tf.keras.utils.register_keras_serializable(package="Addons")
class LiftedStructLoss(LossFunctionWrapper):
    """Computes the lifted structured loss.

    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than any negative distances (between
    a pair of embeddings with different labels) in the mini-batch in a way
    that is differentiable with respect to the embedding vectors.
    See: https://arxiv.org/abs/1511.06452.

    Args:
      margin: Float, margin term in the loss definition.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self, margin: FloatTensorLike = 1.0, name: Optional[str] = None, **kwargs
    ):
        super().__init__(
            lifted_struct_loss,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
        )
