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
"""Implements SoftF1 loss."""

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from typeguard import typechecked
from typing import Optional

from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike


@tf.keras.utils.register_keras_serializable(package="Addons")
class SoftF1Loss(LossFunctionWrapper):
    
    r"""Implements the SoftF1 loss function.
    
    SoftF1 loss was discussed in the sigmoidF1 paper
    (https://arxiv.org/pdf/2108.10566.pdf). It is a loss function 
    based on an approximation of the F1 score that is smooth but
    unbounded and non-saturated and gradient of loss is 1. Therefore
    this is called as the UnboundedF1. Works for both
    multi-class and multi-label classification.
    $$
    L_{\overline{F1}} = \frac{2*\overline{tp}}{2* \overline{tp} + \overline{fp} + \overline{fn}} 
    $$
    where 
    $$
    {\widetilde{tp}}= S(\hat{y})\odot y
    $$

    Args:
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is `macro`.
        name: (Optional) String name of the loss instance.

    Returns:
        SoftF1 loss of the whole batch as a Tensor scalar.

    Raises:
        ValueError: If the `average` has values other than
        `['micro', 'macro', 'weighted']`.
        ValueError: If the shape of `sample_weight` is invalid.
        ValueError: If the shape of `Y_true` and `y_pred` arent same.

    `average` parameter behavior:
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted loss is returned.
        weighted: Losses are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    Usage:
    >>> softf1 = tfa.loss.SoftF1Loss()
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> loss= softf1(y_true, y_pred)
    >>> loss
    <tf.Tensor: shape=(), dtype=float32, numpy=0.39710146>

    Usage with `tf.keras` API:
    >>> model = tf.keras.Model()
    >>> model.compile('sgd', loss=tfa.losses.SoftF1Loss())
    """

    @typechecked
    def __init__(
        self,
        from_logits: bool = False,
        average: Optional[str] = 'macro',
        name: str = "soft_f1",
        **kwargs,
    ):
        super().__init__(
            softf1_loss,
            name=name,
            from_logits=from_logits,
            average= average,
        )
        


@tf.function
def softf1_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    from_logits: bool = False,
    average: Optional[str] = 'macro',
    sample_weight=None,
) -> tf.Tensor:
    
    r"""Implements the SoftF1 loss function.
    
    SoftF1 loss was discussed in the sigmoidF1 paper
    (https://arxiv.org/pdf/2108.10566.pdf). It is a loss function 
    based on an approximation of the F1 score that is smooth but
    unbounded and non-saturated and gradient of loss is 1. Therefore
    this is called as the UnboundedF1. Works for both
    multi-class and multi-label classification.
    $$
    L_{\overline{F1}} = \frac{2*\overline{tp}}{2* \overline{tp} + \overline{fp} + \overline{fn}} 
    $$
    where 
    $$
    {\widetilde{tp}}= S(\hat{y})\odot y
    $$

    Args:
        y_true: True targets tensor.
        y_pred: Predicted tensor.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is `macro`.
        name: (Optional) String name of the loss instance.

    Returns:
        SoftF1 loss of the whole batch as a Tensor scalar.
    """

    if average not in ("micro", "macro", "weighted"):
        raise ValueError(
                "Unknown average type. Acceptable values "
                "are: ['micro', 'macro', 'weighted']"
            )

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    y_true.shape.assert_is_compatible_with(y_pred.shape)


    axis=0
    # axis = 0 calculates global mean
    if average is 'micro':
        axis=None


    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred
      
    # find weighted sum of true positive, true negative and false negative
    def _weighted_sum(val, axis, sample_weight=None):
        if sample_weight is not None:
            val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
        return tf.reduce_sum(val, axis= axis)


    # True positive, true negative and false nagative are made continous as opposed to regular f1 score
    true_positive = _weighted_sum(y_pred * y_true, axis, sample_weight)
    false_positive = _weighted_sum(y_pred * (1 - y_true), axis, sample_weight)
    false_negative = _weighted_sum((1 - y_pred) * y_true, axis, sample_weight)
    weights_intermediate= _weighted_sum(y_true, axis, sample_weight)

    soft_f1 = tf.math.divide_no_nan(2*true_positive , 2*true_positive + false_negative + false_positive )

    if average is "weighted":
        weights = tf.math.divide_no_nan(
                weights_intermediate, tf.reduce_sum(weights_intermediate)
            )
        soft_f1 = tf.reduce_sum(soft_f1 * weights)

    else:  # [micro, macro]
        soft_f1 = tf.reduce_mean(soft_f1)

    cost = 1 - soft_f1 

    # compute the final loss and return
    return cost