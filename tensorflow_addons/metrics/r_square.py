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

import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import weights_broadcast_ops

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes


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

     Usage:
     ```python
     actuals = tf.constant([1, 4, 3], dtype=tf.float32)
     preds = tf.constant([2, 4, 4], dtype=tf.float32)
     result = tf.keras.metrics.RSquare()
     result.update_state(actuals, preds)
     print('R^2 score is: ', r1.result().numpy()) # 0.57142866
    ```
    """

    @typechecked
    def __init__(
        self, name: str = "r_square", dtype: AcceptableDTypes = None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.squared_sum = self.add_weight("squared_sum", initializer="zeros")
        self.sum = self.add_weight("sum", initializer="zeros")
        self.res = self.add_weight("residual", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        if sample_weight is None:
            sample_weight = 1
        sample_weight = tf.cast(sample_weight, self._dtype)
        sample_weight = weights_broadcast_ops.broadcast_weights(sample_weight, y_true)

        weighted_y_true = tf.multiply(y_true, sample_weight)
        self.sum.assign_add(tf.reduce_sum(weighted_y_true))
        self.squared_sum.assign_add(tf.reduce_sum(tf.multiply(y_true, weighted_y_true)))
        self.res.assign_add(
            tf.reduce_sum(
                tf.multiply(tf.square(tf.subtract(y_true, y_pred)), sample_weight)
            )
        )
        self.count.assign_add(tf.reduce_sum(sample_weight))

    def result(self):
        mean = self.sum / self.count
        total = self.squared_sum - self.sum * mean
        return 1 - (self.res / total)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.squared_sum.assign(0.0)
        self.sum.assign(0.0)
        self.res.assign(0.0)
        self.count.assign(0.0)
