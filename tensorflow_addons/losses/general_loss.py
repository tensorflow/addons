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
"""Implements the general form of the loss."""

import tensorflow as tf
import numpy as np

from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class GeneralLoss(tf.keras.losses.Loss):
    """Implements the general form of the loss.

    This is the simplest way of using this loss. No parameters will be tuned
    automatically, it's just a simple function that takes in parameters (likely
    hand-tuned ones) and return a loss. This implements the rho(x, \alpha, c)
    function described in "A General and Adaptive Robust Loss Function",
    Jonathan T. Barron, https://arxiv.org/abs/1701.03077.

    Usage:

    ```python
    gl = tfa.losses.GeneralLoss()
    y_true = tf.constant([[0.97], [0.91], [0.03]], dtype=tf.dtypes.float64)
    y_pred = tf.constant([[1.0], [1.0], [0.0]], dtype=tf.dtypes.float64)
    alpha = tf.constant(2.0, dtype=tf.dtypes.float64)
    scale = tf.constant(1.0, dtype=tf.dtypes.float64)
    loss = gl(y_true, y_pred, alpha=alpha, scale=scale)
    print('Loss: ', loss.numpy())  # Loss: [[0.00045]
                                            [0.00405]
                                            [0.00045]]
    ```

    Usage with `tf.keras` API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tf.keras.losses.GeneralLoss())
    ```

    Args:
      y_true: Actual targets tensor.
      y_pred: Prediction tensor.
      alpha: The shape parameter of the loss (\alpha in the paper), where more
        negative values produce a loss with more robust behavior (outliers
        "cost" less), and more positive values produce a loss with less robust
        behavior (outliers are penalized more heavily). Alpha can be any value
        in [-infinity, infinity], but the gradient of the loss with respect to
        `alpha` is 0 at -infinity, infinity, 0, and 2. Must be a tensorflow
        tensor or numpy array of floats with the same precision as `y_true`
        and `y_pred`. Default value of `alpha` is 2.
        Varying `alpha` allows for smooth interpolation between a number of
        discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
      scale: The `scale` parameter of the loss. When |y_true - y_pred| < scale,
        the loss is an L2-like quadratic bowl, and when
        |y_true - y_pred| > scale the loss function takes on a different shape
        according to `alpha`. Must be a tensorflow tensor or numpy array of
        single-precision floats. Default value for `scale` is 1.
      approximate: a bool, where if True, this function returns an approximate
        and faster form of the loss, as described in the appendix of the paper.
        This approximation holds well everywhere except as residual and
        `alpha` approach zero. By default, `approximate` is set to False.
      epsilon: A float that determines how inaccurate the "approximate" version
        of the loss will be. Larger values are less accurate but more
        numerically stable. Must be greater than single-precision machine
        epsilon. Default value of `epsilon` is 1e-6.

    Returns:
      The losses for each element of residual, in the same shape as residual.
      This is returned as a TensorFlow graph node of single precision floats.

    Raises:
      ValueError: If `epsilon` is less than or equal to single-precision
        machine epsilon. And if `scale` is less than or equal to 0.
      TypeError: If `alpha` or `scale` is of different dtype than `y_true`
        and `y_pred`.
    """

    @typechecked
    def __init__(
        self,
        from_logits: bool = False,
        alpha: FloatTensorLike = 2.0,
        scale: FloatTensorLike = 1.0,
        approximate: bool = False,
        epsilon: FloatTensorLike = 1e-6,
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "general_loss",
    ):
        super().__init__(name=name, reduction=reduction)
        self.from_logits = from_logits
        self.alpha = alpha
        self.scale = scale
        self.approximate = approximate
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        return general_loss(
            y_true,
            y_pred,
            alpha=self.alpha,
            scale=self.scale,
            approximate=self.approximate,
            epsilon=self.epsilon,
            from_logits=self.from_logits,
        )

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
            "alpha": self.alpha,
            "scale": self.scale,
            "approximate": self.approximate,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def general_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = tf.constant(2.0, dtype=tf.dtypes.float64),
    scale: FloatTensorLike = tf.constant(1.0, dtype=tf.dtypes.float64),
    approximate: bool = False,
    epsilon: FloatTensorLike = 1e-6,
    from_logits: bool = False,
) -> tf.Tensor:
    """Implements the general form of the loss.

    This is the simplest way of using this loss. No parameters will be tuned
    automatically, it's just a simple function that takes in parameters (likely
    hand-tuned ones) and return a loss. This implements the rho(x, \alpha, c)
    function described in "A General and Adaptive Robust Loss Function",
    Jonathan T. Barron, https://arxiv.org/abs/1701.03077.

    Usage:

    ```python
    gl = tfa.losses.GeneralLoss()
    y_true = tf.constant([[0.97], [0.91], [0.03]], dtype=tf.dtypes.float64)
    y_pred = tf.constant([[1.0], [1.0], [0.0]], dtype=tf.dtypes.float64)
    alpha = tf.constant(2.0, dtype=tf.dtypes.float64)
    scale = tf.constant(1.0, dtype=tf.dtypes.float64)
    loss = gl(y_true, y_pred, alpha=alpha, scale=scale)
    print('Loss: ', loss.numpy())  # Loss: [[0.00045]
                                            [0.00405]
                                            [0.00045]]
    ```

    Usage with `tf.keras` API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tf.keras.losses.GeneralLoss())
    ```

    Args:
      y_true: Actual targets tensor.
      y_pred: Prediction tensor.
      alpha: The shape parameter of the loss (\alpha in the paper), where more
        negative values produce a loss with more robust behavior (outliers
        "cost" less), and more positive values produce a loss with less robust
        behavior (outliers are penalized more heavily). Alpha can be any value
        in [-infinity, infinity], but the gradient of the loss with respect to
        `alpha` is 0 at -infinity, infinity, 0, and 2. Must be a tensorflow
        tensor or numpy array of floats with the same precision as `y_true`
        and `y_pred`. Default value of `alpha` is 2.
        Varying `alpha` allows for smooth interpolation between a number of
        discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
      scale: The `scale` parameter of the loss. When |y_true - y_pred| < scale,
        the loss is an L2-like quadratic bowl, and when
        |y_true - y_pred| > scale the loss function takes on a different shape
        according to `alpha`. Must be a tensorflow tensor or numpy array of
        single-precision floats. Default value for `scale` is 1.
      approximate: a bool, where if True, this function returns an approximate
        and faster form of the loss, as described in the appendix of the paper.
        This approximation holds well everywhere except as residual and
        `alpha` approach zero. By default, `approximate` is set to False.
      epsilon: A float that determines how inaccurate the "approximate" version
        of the loss will be. Larger values are less accurate but more
        numerically stable. Must be greater than single-precision machine
        epsilon. Default value of `epsilon` is 1e-6.

    Returns:
      The losses for each element of residual, in the same shape as residual.
      This is returned as a TensorFlow graph node of single precision floats.

    Raises:
      ValueError: If `epsilon` is less than or equal to single-precision
        machine epsilon. And if `scale` is less than or equal to 0.
      TypeError: If `alpha` or `scale` is of different dtype than `y_true`
        and `y_pred`.
    """
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tesnor(y_pred)
    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        y_pred = tf.sigmoid(y_pred)

    # Computing residual x from y_true and y_pred.
    x = y_true - y_pred

    # `scale` and `alpha` must have the same type as `x`.
    float_dtype = x.dtype
    tf.debugging.assert_type(scale, float_dtype)
    tf.debugging.assert_type(alpha, float_dtype)

    # `scale` must be > 0.
    assert_ops = [tf.Assert(tf.reduce_all(tf.greater(scale, 0.0)), [scale])]

    with tf.control_dependencies(assert_ops):
        # Broadcast `alpha` and `scale` to have the same shape as `x`.
        alpha = tf.broadcast_to(alpha, tf.shape(x))
        scale = tf.broadcast_to(scale, tf.shape(x))

        if approximate:
            # `epsilon` must be greater than single-precision machine epsilon.
            if epsilon <= np.finfo(np.float32).eps:
                raise ValueError(
                    "The value of epsilon must be greater than",
                    "single-precision machine epsilon",
                )
            # Compute an approximate form of the loss which is faster.
            # But innacurate when x and alpha are near zero.
            b = tf.abs(alpha - tf.cast(2.0, float_dtype)) + epsilon
            d = tf.where(tf.greater_equal(alpha, 0.0), alpha + epsilon, alpha - epsilon)
            loss = (b / d) * (tf.pow(tf.square(x / scale) / b + 1.0, 0.5 * d) - 1.0)
        else:
            # Compute the exact loss.

            # This will be used repeatedly.
            squared_scaled_x = tf.square(x / scale)

            # The loss when alpha = 2.
            loss_two = 0.5 * squared_scaled_x
            # The loss when alpha = 0.
            loss_zero = tf.math.log1p(
                tf.minimum(
                    0.5 * squared_scaled_x, tf.cast(3e37, squared_scaled_x.dtype)
                )
            )
            # The loss when alpha = -infinity.
            loss_neginf = -tf.math.expm1(-0.5 * squared_scaled_x)
            # The loss when alpha = +infinity.
            loss_posinf = tf.math.expm1(
                tf.minimum(
                    0.5 * squared_scaled_x, tf.cast(87.5, squared_scaled_x.dtype)
                )
            )

            # The loss when not in one of the above special cases.
            machine_epsilon = tf.cast(np.finfo(np.float32).eps, float_dtype)
            # Clamp |2-alpha| to be >= machine epsilon.
            # So that it's safe to divide by.
            beta_safe = tf.maximum(machine_epsilon, tf.abs(alpha - 2.0))
            # Clamp |alpha| to be >= machine epsilon.
            # So that it's safe to divide by.
            alpha_safe = tf.where(
                tf.greater_equal(alpha, 0.0), tf.ones_like(alpha), -tf.ones_like(alpha)
            ) * tf.maximum(machine_epsilon, tf.abs(alpha))
            loss_otherwise = (beta_safe / alpha_safe) * (
                tf.pow(squared_scaled_x / beta_safe + 1.0, 0.5 * alpha) - 1.0
            )

            # Select which of the cases of the loss to return.
            loss = tf.where(
                tf.equal(alpha, -tf.cast(float("inf"), float_dtype)),
                loss_neginf,
                tf.where(
                    tf.equal(alpha, 0.0),
                    loss_zero,
                    tf.where(
                        tf.equal(alpha, 2.0),
                        loss_two,
                        tf.where(
                            tf.equal(alpha, tf.cast(float("inf"), float_dtype)),
                            loss_posinf,
                            loss_otherwise,
                        ),
                    ),
                ),
            )

        return loss
