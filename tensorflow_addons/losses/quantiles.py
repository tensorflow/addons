# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements quantiles losses."""

import tensorflow as tf
from typeguard import typechecked
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import TensorLike, FloatTensorLike


@tf.function
@tf.keras.utils.register_keras_serializable(package="Addons")
def pinball_loss(
    y_true: TensorLike, y_pred: TensorLike, tau: FloatTensorLike = 0.5
) -> tf.Tensor:
    """Computes the pinball loss between `y_true` and `y_pred`.

    `loss = maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))`

    In the context of regression this, loss yields an estimator of the tau
    conditional quantile.

    See: https://en.wikipedia.org/wiki/Quantile_regression

    Usage:
    ```python
    loss = pinball_loss([0., 0., 1., 1.], [1., 1., 1., 0.], tau=.1)

    # loss = max(0.1 * (y_true - y_pred), (0.1 - 1) * (y_true - y_pred))
    #      = (0.9 + 0.9 + 0 + 0.1) / 4

    print('Loss: ', loss.numpy())  # Loss: 0.475
    ```

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
      tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
        shape = `[d0,..., dn]`.  It defines the slope of the pinball loss. In
        the context of quantile regression, the value of tau determines the
        conditional quantile level. When tau = 0.5, this amounts to l1
        regression, an estimator of the conditional median (0.5 quantile).

    Returns:
        pinball_loss: 1-D float `Tensor` with shape [batch_size].

    References:
      - https://en.wikipedia.org/wiki/Quantile_regression
      - https://projecteuclid.org/download/pdfview_1/euclid.bj/1297173840
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Broadcast the pinball slope along the batch dimension
    tau = tf.expand_dims(tf.cast(tau, y_pred.dtype), 0)
    one = tf.cast(1, tau.dtype)

    delta_y = y_true - y_pred
    pinball = tf.math.maximum(tau * delta_y, (tau - one) * delta_y)
    return tf.reduce_mean(pinball, axis=-1)


@tf.keras.utils.register_keras_serializable(package="Addons")
class PinballLoss(LossFunctionWrapper):
    """Computes the pinball loss between `y_true` and `y_pred`.

    `loss = maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))`

    In the context of regression, this loss yields an estimator of the tau
    conditional quantile.

    See: https://en.wikipedia.org/wiki/Quantile_regression

    Usage:
    ```python
    pinball = tfa.losses.PinballLoss(tau=.1)
    loss = pinball([0., 0., 1., 1.], [1., 1., 1., 0.])

    # loss = max(0.1 * (y_true - y_pred), (0.1 - 1) * (y_true - y_pred))
    #      = (0.9 + 0.9 + 0 + 0.1) / 4

    print('Loss: ', loss.numpy())  # Loss: 0.475
    ```

    Usage with the `compile` API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tfa.losses.PinballLoss(tau=.1))
    ```

    Args:
      tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
        shape = `[d0,..., dn]`.  It defines the slope of the pinball loss. In
        the context of quantile regression, the value of tau determines the
        conditional quantile level. When tau = 0.5, this amounts to l1
        regression, an estimator of the conditional median (0.5 quantile).
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`.
        When used with `tf.distribute.Strategy`, outside of built-in training
        loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
        `SUM_OVER_BATCH_SIZE` will raise an error. Please see
        https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
        for more details on this.
      name: Optional name for the op.

    References:
      - https://en.wikipedia.org/wiki/Quantile_regression
      - https://projecteuclid.org/download/pdfview_1/euclid.bj/1297173840
    """

    @typechecked
    def __init__(
        self,
        tau: FloatTensorLike = 0.5,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = "pinball_loss",
    ):
        super().__init__(pinball_loss, reduction=reduction, name=name, tau=tau)
