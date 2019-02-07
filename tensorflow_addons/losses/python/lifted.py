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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import losses
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses_impl

def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(
            math_ops.square(feature),
            axis=[1],
            keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(
                feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared,
                                                  0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances,
        math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(
        pairwise_distances) - array_ops.diag(array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances,
                                           mask_offdiagonals)
    return pairwise_distances

def lifted_struct_loss(labels, embeddings, margin=1.0):
  """Computes the lifted structured loss.
  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should not
      be l2 normalized.
    margin: Float, margin term in the loss definition.
  Returns:
    lifted_loss: tf.float32 scalar.
  """
  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  # Build pairwise squared distance matrix.
  pairwise_distances = pairwise_distance(embeddings)

  # Build pairwise binary adjacency matrix.
  adjacency = math_ops.equal(labels, array_ops.transpose(labels))
  # Invert so we can select negatives only.
  adjacency_not = math_ops.logical_not(adjacency)

  batch_size = array_ops.size(labels)

  diff = margin - pairwise_distances
  mask = math_ops.cast(adjacency_not, dtype=dtypes.float32)
  # Safe maximum: Temporarily shift negative distances
  #   above zero before taking max.
  #     this is to take the max only among negatives.
  row_minimums = math_ops.reduce_min(diff, 1, keepdims=True)
  row_negative_maximums = math_ops.reduce_max(
      math_ops.multiply(diff - row_minimums, mask), 1,
      keepdims=True) + row_minimums

  # Compute the loss.
  # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
  #   where m_i is the max of alpha - negative D_i's.
  # This matches the Caffe loss layer implementation at:
  #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp  # pylint: disable=line-too-long

  max_elements = math_ops.maximum(
      row_negative_maximums, array_ops.transpose(row_negative_maximums))
  diff_tiled = array_ops.tile(diff, [batch_size, 1])
  mask_tiled = array_ops.tile(mask, [batch_size, 1])
  max_elements_vect = array_ops.reshape(
      array_ops.transpose(max_elements), [-1, 1])

  loss_exp_left = array_ops.reshape(
      math_ops.reduce_sum(
          math_ops.multiply(
              math_ops.exp(diff_tiled - max_elements_vect), mask_tiled),
          1,
          keepdims=True), [batch_size, batch_size])

  loss_mat = max_elements + math_ops.log(
      loss_exp_left + array_ops.transpose(loss_exp_left))
  # Add the positive distance.
  loss_mat += pairwise_distances

  mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

  # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
  num_positives = math_ops.reduce_sum(mask_positives) / 2.0

  lifted_loss = math_ops.truediv(
      0.25 * math_ops.reduce_sum(
          math_ops.square(
              math_ops.maximum(
                  math_ops.multiply(loss_mat, mask_positives), 0.0))),
      num_positives,
      name='liftedstruct_loss')
  return lifted_loss

class LiftedStructLoss(losses.LossFunctionWrapper):
    """Computes the lifted structured loss.
    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than any negative distances (between a
    pair of embeddings with different labels) in the mini-batch in a way
    that is differentiable with respect to the embedding vectors.
    See: https://arxiv.org/abs/1511.06452.
    Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
        multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should not
        be l2 normalized.
        margin: Float, margin term in the loss definition.
    Returns:
        lifted_loss: tf.float32 scalar.
    """
    
    def __init__(self, margin=1.0, name=None):
        super(LiftedStructLoss, self).__init__(
            lifted_struct_loss,
            name=name,
            reduction=losses_impl.ReductionV2.NONE,
            margin=margin)
