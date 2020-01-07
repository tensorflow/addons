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
"""Implements pinball loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras.backend as K
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Addons')
def pinball_loss(y_true, y_pred, tau):
    """Computes the pinball loss between `y_true` and `y_pred`.

    Usage:
    ```python
    loss = pinball_loss([0., 0., 1., 1.], [1., 1., 1., 0.], tau=.1)
    print('Loss: ', loss.numpy())  # Loss:
    ```
    Args:
    tau: a float between 0 and 1 the slope of the pinball loss. In the context
    of quantile regression, the value of tau determines the conditional
    quantile level.
    """
    delta_y = y_true - y_pred
    return K.maximum(tau * delta_y, (tau - 1) * delta_y)


@tf.keras.utils.register_keras_serializable(package='Addons')
class PinballLoss(tf.keras.losses.Loss):
    """Computes the pinball loss between `y_true` and `y_pred`.

    Usage:
    ```python
    pinball = tfa.losses.PinballLoss(tau=.1, axis=1)
    loss = pinball([0., 0., 1., 1.], [1., 1., 1., 0.])
    print('Loss: ', loss.numpy())  # Loss:
    ```
    Usage with the `compile` API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tfa.losses.PinballLoss(tau=.9))
    ```
    Args:
    tau: a float between 0 and 1 the slope of the pinball loss. In the context
      of quantile regression, the value of tau determines the conditional
      quantile level.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`.
      When used with `tf.distribute.Strategy`, outside of built-in training
      loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
      `SUM_OVER_BATCH_SIZE` will raise an error. Please see
      https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
      for more details on this.
    name: Optional name for the op.
    """

    def __init__(self,
                 tau=.5,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='pinball_loss'):
        super(PinballLoss, self).__init__(reduction=reduction, name=name)
        self.tau = tau

    def call(self, y_true, y_pred):
        return pinball_loss(y_true, y_pred, self.tau)

    def get_config(self):
        config = {
            'tau': self.tau,
        }
        base_config = super(PinballLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
