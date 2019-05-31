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
"""Implements Focal loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    """Implements the focal loss function.

    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.

    Usage:

    ```python
    fl = tfa.losses.SigmoidFocalCrossEntropy()
    loss = fl(
      [[0.97], [0.91], [0.03]],
      [[1], [1], [0])
    print('Loss: ', loss.numpy())  # Loss: [[0.03045921]
                                            [0.09431068]
                                            [0.31471074]
    ```
    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tf.keras.losses.SigmoidFocalCrossEntropy())
    ```

    Args
      alpha: balancing factor, default value is 0.25
      gamma: modulating factor, default value is 2.0

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
          shape as `y_true`; otherwise, it is scalar.

    Raises:
        ValueError: If the shape of `sample_weight` is invalid or value of
          `gamma` is less than zero
    """

    def __init__(self,
                 from_logits=False,
                 alpha=0.25,
                 gamma=2.0,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='sigmoid_focal_crossentropy'):
        super(SigmoidFocalCrossEntropy, self).__init__(
            name=name, reduction=reduction)

        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        return sigmoid_focal_crossentropy(
            y_true,
            y_pred,
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=self.from_logits)

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        base_config = super(SigmoidFocalCrossEntropy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_utils.register_keras_custom_object
@tf.function
def sigmoid_focal_crossentropy(y_true,
                               y_pred,
                               alpha=0.25,
                               gamma=2.0,
                               from_logits=False):
    """
    Args
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the 
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the binary cross_entropy
    bce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        y_pred = K.sigmoid(y_pred)
    else:
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = 1
    modulating_factor = 1

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = K.pow((1 - p_t), gamma)

    # compute the final loss and return
    return K.mean(
        alpha_factor * modulating_factor * bce, axis=-1, keepdims=True)
