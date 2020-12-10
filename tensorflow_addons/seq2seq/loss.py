# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Loss functions for sequence models."""

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike

from typeguard import typechecked
from typing import Callable, Optional


def sequence_loss(
    logits: TensorLike,
    targets: TensorLike,
    weights: TensorLike,
    average_across_timesteps: bool = True,
    average_across_batch: bool = True,
    sum_over_timesteps: bool = False,
    sum_over_batch: bool = False,
    softmax_loss_function: Optional[Callable] = None,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Computes the weighted cross-entropy loss for a sequence of logits.

    Depending on the values of `average_across_timesteps` /
    `sum_over_timesteps` and `average_across_batch` / `sum_over_batch`, the
    return Tensor will have rank 0, 1, or 2 as these arguments reduce the
    cross-entropy at each target, which has shape
    `[batch_size, sequence_length]`, over their respective dimensions. For
    example, if `average_across_timesteps` is `True` and `average_across_batch`
    is `False`, then the return Tensor will have shape `[batch_size]`.

    Note that `average_across_timesteps` and `sum_over_timesteps` cannot be
    True at same time. Same for `average_across_batch` and `sum_over_batch`.

    The recommended loss reduction in tf 2.0 has been changed to sum_over,
    instead of weighted average. User are recommend to use `sum_over_timesteps`
    and `sum_over_batch` for reduction.

    Args:
      logits: A Tensor of shape
        `[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
        The logits correspond to the prediction across all classes at each
        timestep.
      targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
        int. The target represents the true class at each timestep.
      weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
        float. `weights` constitutes the weighting of each prediction in the
        sequence. When using `weights` as masking, set all valid timesteps to 1
        and all padded timesteps to 0, e.g. a mask returned by
        `tf.sequence_mask`.
      average_across_timesteps: If set, sum the cost across the sequence
        dimension and divide the cost by the total label weight across
        timesteps.
      average_across_batch: If set, sum the cost across the batch dimension and
        divide the returned cost by the batch size.
      sum_over_timesteps: If set, sum the cost across the sequence dimension
        and divide the size of the sequence. Note that any element with 0
        weights will be excluded from size calculation.
      sum_over_batch: if set, sum the cost across the batch dimension and
        divide the total cost by the batch size. Not that any element with 0
        weights will be excluded from size calculation.
      softmax_loss_function: Function (labels, logits) -> loss-batch
        to be used instead of the standard softmax (the default if this is
        None). **Note that to avoid confusion, it is required for the function
        to accept named arguments.**
      name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
      A float Tensor of rank 0, 1, or 2 depending on the
      `average_across_timesteps` and `average_across_batch` arguments. By
      default, it has rank 0 (scalar) and is the weighted average cross-entropy
      (log-perplexity) per symbol.

    Raises:
      ValueError: logits does not have 3 dimensions or targets does not have 2
                  dimensions or weights does not have 2 dimensions.
    """
    if len(logits.shape) != 3:
        raise ValueError(
            "Logits must be a [batch_size x sequence_length x logits] tensor"
        )

    targets_rank = len(targets.shape)
    if targets_rank != 2 and targets_rank != 3:
        raise ValueError(
            "Targets must be either a [batch_size x sequence_length] tensor "
            + "where each element contains the labels' index"
            + "or a [batch_size x sequence_length x num_classes] tensor "
            + "where the third axis is a one-hot representation of the labels"
        )

    if len(weights.shape) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] tensor")

    if average_across_timesteps and sum_over_timesteps:
        raise ValueError(
            "average_across_timesteps and sum_over_timesteps cannot "
            "be set to True at same time."
        )
    if average_across_batch and sum_over_batch:
        raise ValueError(
            "average_across_batch and sum_over_batch cannot be set "
            "to True at same time."
        )
    if average_across_batch and sum_over_timesteps:
        raise ValueError(
            "average_across_batch and sum_over_timesteps cannot be set "
            "to True at same time because of ambiguous order."
        )
    if sum_over_batch and average_across_timesteps:
        raise ValueError(
            "sum_over_batch and average_across_timesteps cannot be set "
            "to True at same time because of ambiguous order."
        )
    with tf.name_scope(name or "sequence_loss"):
        num_classes = tf.shape(input=logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        if softmax_loss_function is None:
            if targets_rank == 2:
                targets = tf.reshape(targets, [-1])
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits_flat
                )
            else:
                targets = tf.reshape(targets, [-1, num_classes])
                crossent = tf.nn.softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits_flat
                )
        else:
            targets = tf.reshape(targets, [-1])
            crossent = softmax_loss_function(labels=targets, logits=logits_flat)
        crossent *= tf.reshape(weights, [-1])
        if average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(input_tensor=crossent)
            total_size = tf.reduce_sum(input_tensor=weights)
            crossent = tf.math.divide_no_nan(crossent, total_size)
        elif sum_over_timesteps and sum_over_batch:
            crossent = tf.reduce_sum(input_tensor=crossent)
            total_count = tf.cast(tf.math.count_nonzero(weights), crossent.dtype)
            crossent = tf.math.divide_no_nan(crossent, total_count)
        else:
            crossent = tf.reshape(crossent, tf.shape(input=logits)[0:2])
            if average_across_timesteps or average_across_batch:
                reduce_axis = [0] if average_across_batch else [1]
                crossent = tf.reduce_sum(input_tensor=crossent, axis=reduce_axis)
                total_size = tf.reduce_sum(input_tensor=weights, axis=reduce_axis)
                crossent = tf.math.divide_no_nan(crossent, total_size)
            elif sum_over_timesteps or sum_over_batch:
                reduce_axis = [0] if sum_over_batch else [1]
                crossent = tf.reduce_sum(input_tensor=crossent, axis=reduce_axis)
                total_count = tf.cast(
                    tf.math.count_nonzero(weights, axis=reduce_axis),
                    dtype=crossent.dtype,
                )
                crossent = tf.math.divide_no_nan(crossent, total_count)
        return crossent


class SequenceLoss(tf.keras.losses.Loss):
    """Weighted cross-entropy loss for a sequence of logits."""

    @typechecked
    def __init__(
        self,
        average_across_timesteps: bool = False,
        average_across_batch: bool = False,
        sum_over_timesteps: bool = True,
        sum_over_batch: bool = True,
        softmax_loss_function: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.average_across_timesteps = average_across_timesteps
        self.average_across_batch = average_across_batch
        self.sum_over_timesteps = sum_over_timesteps
        self.sum_over_batch = sum_over_batch
        self.softmax_loss_function = softmax_loss_function

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Override the parent __call__ to have a customized reduce
        behavior."""
        return sequence_loss(
            y_pred,
            y_true,
            sample_weight,
            average_across_timesteps=self.average_across_timesteps,
            average_across_batch=self.average_across_batch,
            sum_over_timesteps=self.sum_over_timesteps,
            sum_over_batch=self.sum_over_batch,
            softmax_loss_function=self.softmax_loss_function,
            name=self.name,
        )

    def call(self, y_true, y_pred):
        # Skip this method since the __call__ contains real implementation.
        pass
