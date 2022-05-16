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
"""Implements R^2 scores."""
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import weights_broadcast_ops

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes


_VALID_MULTIOUTPUT = {"raw_values", "uniform_average", "variance_weighted"}


def _reduce_average(
    input_tensor: tf.Tensor, axis=None, keepdims=False, weights=None
) -> tf.Tensor:
    """Computes the (weighted) mean of elements across dimensions of a tensor."""
    if weights is None:
        return tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims)

    weighted_sum = tf.reduce_sum(weights * input_tensor, axis=axis, keepdims=keepdims)
    sum_of_weights = tf.reduce_sum(weights, axis=axis, keepdims=keepdims)
    average = weighted_sum / sum_of_weights
    return average


@tf.keras.utils.register_keras_serializable(package="Addons")
class RSquare(Metric):
    """Compute R^2 score.

    This is also called the [coefficient of determination
    ](https://en.wikipedia.org/wiki/Coefficient_of_determination).
    It tells how close are data to the fitted regression line.

    - Highest score can be 1.0 and it indicates that the predictors
        perfectly accounts for variation in the target.
    - Score 0.0 indicates that the predictors do not
        account for variation in the target.
    - It can also be negative if the model is worse.

    The sample weighting for this metric implementation mimics the
    behaviour of the [scikit-learn implementation
    ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
    of the same metric.

    Can also calculate the Adjusted R2 Score.

    Args:
        multioutput: `string`, the reduce method for scores.
            Should be one of `["raw_values", "uniform_average", "variance_weighted"]`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        num_regressors: (Optional) Number of indepedent regressors used (Adjusted R2).
            Defaults to zero(standard R2 score).

    Usage:

    >>> y_true = np.array([1, 4, 3], dtype=np.float32)
    >>> y_pred = np.array([2, 4, 4], dtype=np.float32)
    >>> metric = tfa.metrics.r_square.RSquare()
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    0.57142854
    """

    @typechecked
    def __init__(
        self,
        name: str = "r_square",
        dtype: AcceptableDTypes = None,
        multioutput: str = "uniform_average",
        num_regressors: tf.int32 = 0,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if "y_shape" in kwargs:
            warnings.warn(
                "y_shape has been removed, because it's automatically derived,"
                "and will be deprecated in Addons 0.18.",
                DeprecationWarning,
            )

        if multioutput not in _VALID_MULTIOUTPUT:
            raise ValueError(
                "The multioutput argument must be one of {}, but was: {}".format(
                    _VALID_MULTIOUTPUT, multioutput
                )
            )
        self.multioutput = multioutput
        self.num_regressors = num_regressors
        self.num_samples = self.add_weight(name="num_samples", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        if not hasattr(self, "squared_sum"):
            self.squared_sum = self.add_weight(
                name="squared_sum",
                shape=y_true.shape[1:],
                initializer="zeros",
                dtype=self._dtype,
            )
        if not hasattr(self, "sum"):
            self.sum = self.add_weight(
                name="sum",
                shape=y_true.shape[1:],
                initializer="zeros",
                dtype=self._dtype,
            )
        if not hasattr(self, "res"):
            self.res = self.add_weight(
                name="residual",
                shape=y_true.shape[1:],
                initializer="zeros",
                dtype=self._dtype,
            )
        if not hasattr(self, "count"):
            self.count = self.add_weight(
                name="count",
                shape=y_true.shape[1:],
                initializer="zeros",
                dtype=self._dtype,
            )

        y_true = tf.cast(y_true, dtype=self._dtype)
        y_pred = tf.cast(y_pred, dtype=self._dtype)
        if sample_weight is None:
            sample_weight = 1
        sample_weight = tf.cast(sample_weight, dtype=self._dtype)
        sample_weight = weights_broadcast_ops.broadcast_weights(
            weights=sample_weight, values=y_true
        )

        weighted_y_true = y_true * sample_weight
        self.sum.assign_add(tf.reduce_sum(weighted_y_true, axis=0))
        self.squared_sum.assign_add(tf.reduce_sum(y_true * weighted_y_true, axis=0))
        self.res.assign_add(
            tf.reduce_sum((y_true - y_pred) ** 2 * sample_weight, axis=0)
        )
        self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))
        self.num_samples.assign_add(tf.size(y_true))

    def result(self) -> tf.Tensor:
        mean = self.sum / self.count
        total = self.squared_sum - self.sum * mean
        raw_scores = 1 - (self.res / total)
        raw_scores = tf.where(tf.math.is_inf(raw_scores), 0.0, raw_scores)

        if self.multioutput == "raw_values":
            r2_score = raw_scores
        elif self.multioutput == "uniform_average":
            r2_score = tf.reduce_mean(raw_scores)
        elif self.multioutput == "variance_weighted":
            r2_score = _reduce_average(raw_scores, weights=total)
        else:
            raise RuntimeError(
                "The multioutput attribute must be one of {}, but was: {}".format(
                    _VALID_MULTIOUTPUT, self.multioutput
                )
            )

        if self.num_regressors < 0:
            raise ValueError(
                "num_regressors parameter should be greater than or equal to zero"
            )

        if self.num_regressors != 0:
            if self.num_regressors > self.num_samples - 1:
                UserWarning(
                    "More independent predictors than datapoints in adjusted r2 score. Falls back to standard r2 "
                    "score."
                )
            elif self.num_regressors == self.num_samples - 1:
                UserWarning(
                    "Division by zero in adjusted r2 score. Falls back to standard r2 score."
                )
            else:
                n = tf.cast(self.num_samples, dtype=tf.float32)
                p = tf.cast(self.num_regressors, dtype=tf.float32)

                num = tf.multiply(tf.subtract(1.0, r2_score), tf.subtract(n, 1.0))
                den = tf.subtract(tf.subtract(n, p), 1.0)
                r2_score = tf.subtract(1.0, tf.divide(num, den))

        return r2_score

    def reset_state(self) -> None:
        # The state of the metric will be reset at the start of each epoch.
        K.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()

    def get_config(self):
        config = {
            "multioutput": self.multioutput,
        }
        base_config = super().get_config()
        return {**base_config, **config}
