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

import tensorflow as tf
from tensorflow_addons.activations.sparsemax import sparsemax

from tensorflow_addons.utils.types import TensorLike
from typeguard import typechecked
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
def sparsemax_loss(
    logits: TensorLike,
    sparsemax: TensorLike,
    labels: TensorLike,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Sparsemax loss function [1].

    Computes the generalized multi-label classification loss for the sparsemax
    function. The implementation is a reformulation of the original loss
    function such that it uses the sparsemax properbility output instead of the
    internal \tau variable. However, the output is identical to the original
    loss function.

    [1]: https://arxiv.org/abs/1602.02068

    Args:
      logits: A `Tensor`. Must be one of the following types: `float32`,
        `float64`.
      sparsemax: A `Tensor`. Must have the same type as `logits`.
      labels: A `Tensor`. Must have the same type as `logits`.
      name: A name for the operation (optional).
    Returns:
      A `Tensor`. Has the same type as `logits`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")
    sparsemax = tf.convert_to_tensor(sparsemax, name="sparsemax")
    labels = tf.convert_to_tensor(labels, name="labels")

    # In the paper, they call the logits z.
    # A constant can be substracted from logits to make the algorithm
    # more numerically stable in theory. However, there are really no major
    # source numerical instability in this algorithm.
    z = logits

    # sum over support
    # Use a conditional where instead of a multiplication to support z = -inf.
    # If z = -inf, and there is no support (sparsemax = 0), a multiplication
    # would cause 0 * -inf = nan, which is not correct in this case.
    sum_s = tf.where(
        tf.math.logical_or(sparsemax > 0, tf.math.is_nan(sparsemax)),
        sparsemax * (z - 0.5 * sparsemax),
        tf.zeros_like(sparsemax),
    )

    # - z_k + ||q||^2
    q_part = labels * (0.5 * labels - z)
    # Fix the case where labels = 0 and z = -inf, where q_part would
    # otherwise be 0 * -inf = nan. But since the lables = 0, no cost for
    # z = -inf should be consideredself.
    # The code below also coveres the case where z = inf. Howeverm in this
    # caose the sparsemax will be nan, which means the sum_s will also be nan,
    # therefor this case doesn't need addtional special treatment.
    q_part_safe = tf.where(
        tf.math.logical_and(tf.math.equal(labels, 0), tf.math.is_inf(z)),
        tf.zeros_like(z),
        q_part,
    )

    return tf.math.reduce_sum(sum_s + q_part_safe, axis=1)


@tf.function
@tf.keras.utils.register_keras_serializable(package="Addons")
def sparsemax_loss_from_logits(
    y_true: TensorLike, logits_pred: TensorLike
) -> tf.Tensor:
    y_pred = sparsemax(logits_pred)
    loss = sparsemax_loss(logits_pred, y_pred, y_true)
    return loss


@tf.keras.utils.register_keras_serializable(package="Addons")
class SparsemaxLoss(tf.keras.losses.Loss):
    """Sparsemax loss function.

    Computes the generalized multi-label classification loss for the sparsemax
    function.

    Because the sparsemax loss function needs both the properbility output and
    the logits to compute the loss value, `from_logits` must be `True`.

    Because it computes the generalized multi-label loss, the shape of both
    `y_pred` and `y_true` must be `[batch_size, num_classes]`.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. Default
        is `True`, meaning `y_pred` is the logits.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `SUM_OVER_BATCH_SIZE`.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "sparsemax_loss",
    ):
        if from_logits is not True:
            raise ValueError("from_logits must be True")

        super().__init__(name=name, reduction=reduction)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return sparsemax_loss_from_logits(y_true, y_pred)

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
        }
        base_config = super().get_config()
        return {**base_config, **config}
