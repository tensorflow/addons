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
"""Implements SigmoidF1 loss."""

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from typeguard import typechecked
from typing import Optional

from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike


@tf.keras.utils.register_keras_serializable(package="Addons")
class SigmoidF1Loss(LossFunctionWrapper):
    
    r"""Implements the SigmoidF1 loss function.
    
    SigmoidF1 loss was first introduced in the sigmoidF1 paper
    (https://arxiv.org/pdf/2108.10566.pdf). It is a loss function 
    based on an approximation of the F1 score that is smooth and 
    tractable for stochastic gradient descent, naturally approximates 
    a multilabel metric and estimates label propensities and label 
    counts. Output range is `[0, 1]`. Works for
    both multi-class and multi-label classification.
    $$
    L_{\widetilde{F1}} = \frac{2*\widetilde{tp}}{2* \widetilde{tp} + \widetilde{fp} + \widetilde{fn}} 
    $$
    where 
    $$
    {\widetilde{tp}}= \sum S(\hat{y})\odot y
    $$
    and
    $$
    S(\upsilon ; \beta,  \eta )= \frac{1}{1+ exp(-\beta (\upsilon+ \eta))}
    $$

    Args:
        beta: Determines the gradient at the center of the sigmoid.
            Default value is 0.001.
        eta: Determines the offset of the sigmoid function at the center
            Default value is 1.0.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is `macro`.
        name: (Optional) String name of the loss instance.

    Returns:
        SigmoidF1 loss of the whole batch as a Tensor scalar.

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
    >>> sigmoidf1 = tfa.loss.SigmoidF1Loss()
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> loss= sigmoidf1(y_true, y_pred)
    >>> loss
    <tf.Tensor: shape=(), dtype=float32, numpy=0.45395237>

    Usage with `tf.keras` API:
    >>> model = tf.keras.Model()
    >>> model.compile('sgd', loss=tfa.losses.SigmoidF1Loss())
    """

    @typechecked
    def __init__(
        self,
        from_logits: bool = False,
        beta: FloatTensorLike = 0.001,
        eta: FloatTensorLike = 1.0,
        average: Optional[str] = 'macro',
        name: str = "sigmoid_f1",
        **kwargs,
    ):
        super().__init__(
            sigmoidf1_loss,
            name=name,
            from_logits=from_logits,
            average= average,
            beta= beta,
            eta= eta
        )
        


@tf.function
def sigmoidf1_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    beta: FloatTensorLike = 0.001,
    eta: FloatTensorLike = 1.0,
    from_logits: bool = False,
    average: Optional[str] = 'macro',
    sample_weight=None,
) -> tf.Tensor:
    
    r"""Implements the SigmoidF1 loss function.
    
    SigmoidF1 loss was first introduced in the sigmoidF1 paper
    (https://arxiv.org/pdf/2108.10566.pdf). It is a loss function 
    based on an approximation of the F1 score that is smooth and 
    tractable for stochastic gradient descent, naturally approximates 
    a multilabel metric and estimates label propensities and label 
    counts. Output range is `[0, 1]`. Works for
    both multi-class and multi-label classification.
    $$
    L_{\widetilde{F1}} = \frac{2*\widetilde{tp}}{2* \widetilde{tp} + \widetilde{fp} + \widetilde{fn}} 
    $$
    where 
    $$
    {\widetilde{tp}}= S(\hat{y})\odot y
    $$
    and
    $$
    S(\upsilon ; \beta,  \eta )= \frac{1}{1+ exp(-\beta (\upsilon+ \eta))}
    $$

    Args:
        y_true: True targets tensor.
        y_pred: Predicted tensor.
        beta: Determines the gradient at the center of the sigmoid.
            Default value is 0.001.
        eta: Determines the offset of the sigmoid function at the center
            Default value is 1.0.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is `macro`.
        name: (Optional) String name of the loss instance.

    Returns:
        SigmoidF1 loss of the whole batch as a Tensor scalar.
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

    def _sigmoid_transformation(val, beta, eta):
        beta = tf.cast(beta, dtype=val.dtype)
        eta = tf.cast(eta, dtype=val.dtype)
        return tf.math.divide(1, 1 + tf.math.exp(-1* tf.math.multiply_no_nan(beta, tf.math.add(val, eta))))

    # True positive, true negative and false nagative are made continous as opposed to regular f1 score
    true_positive = _weighted_sum(_sigmoid_transformation(y_pred, beta, eta) * y_true, axis, sample_weight)
    false_positive = _weighted_sum(_sigmoid_transformation(y_pred, beta, eta) * (1 - y_true), axis, sample_weight)
    false_negative = _weighted_sum(_sigmoid_transformation(1 - y_pred, beta, eta) * y_true, axis, sample_weight)
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