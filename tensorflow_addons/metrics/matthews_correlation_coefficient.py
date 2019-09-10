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
"""Matthews Correlation Coefficient Implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.metrics import Metric
import numpy as np


class MatthewsCorrelationCoefficient(Metric):
    """Computes the Matthews Correlation Coefficient.

    The statistic is also known as the phi coefficient.
    The Matthews correlation coefficient (MCC) is used in
    machine learning as a measure of the quality of binary
    and multiclass classifications. It takes into account
    true and false positives and negatives and is generally
    regarded as a balanced measure which can be used even
    if the classes are of very different sizes. The correlation
    coefficient value of MCC is between -1 and +1. A
    coefficient of +1 represents a perfect prediction,
    0 an average random prediction and -1 an inverse
    prediction. The statistic is also known as
    the phi coefficient.

    MCC = (TP * TN) - (FP * FN) /
          ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2)

    Args:
       num_classes : Number of unique classes in the dataset.

    Returns:
       mcc : float (the Matthews correlation coefficient)

    Usage:
    ```python
    actuals = tf.constant([[1.0], [1.0], [1.0], [0.0]],
             dtype=tf.float32)
    preds = tf.constant([[1.0], [0.0], [1.0], [1.0]],
             dtype=tf.float32)
    # Matthews correlation coefficient
    mcc = MatthewsCorrelationCoefficient(num_classes=1)
    output.update_state(actuals, preds)
    print('Matthews correlation coefficient is:',
    output.result().numpy())
    # Matthews correlation coefficient is : -0.33333334

    ```
    """

    def __init__(self,
                 num_classes=None,
                 name='MatthewsCorrelationCoefficient',
                 dtype=tf.float32):
        super(MatthewsCorrelationCoefficient, self).__init__(
            name=name, dtype=dtype)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            'true_positives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=self.dtype)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=self.dtype)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=self.dtype)
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=self.dtype)

    # TODO: sample_weights
    def update_state(self, y_true, y_pred, sample_weights=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        true_positive = tf.math.count_nonzero(y_true * y_pred, 0)
        # predicted sum
        pred_sum = tf.math.count_nonzero(y_pred, 0)
        # Ground truth label sum
        true_sum = tf.math.count_nonzero(y_true, 0)
        false_positive = pred_sum - true_positive
        false_negative = true_sum - true_positive
        true_negative = y_true.get_shape(
        )[0] - true_positive - false_positive - false_negative

        # true positive state_update
        self.true_positives.assign_add(tf.cast(true_positive, self.dtype))
        # false positive state_update
        self.false_positives.assign_add(tf.cast(false_positive, self.dtype))
        # false negative state_update
        self.false_negatives.assign_add(tf.cast(false_negative, self.dtype))
        # true negative state_update
        self.true_negatives.assign_add(tf.cast(true_negative, self.dtype))

    def result(self):
        # numerator
        numerator1 = self.true_positives * self.true_negatives
        numerator2 = self.false_positives * self.false_negatives
        numerator = numerator1 - numerator2
        # denominator
        denominator1 = self.true_positives + self.false_positives
        denominator2 = self.true_positives + self.false_negatives
        denominator3 = self.true_negatives + self.false_positives
        denominator4 = self.true_negatives + self.false_negatives
        denominator = tf.math.sqrt(
            denominator1 * denominator2 * denominator3 * denominator4)
        mcc = tf.math.divide_no_nan(numerator, denominator)
        return mcc

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
        }
        base_config = super(MatthewsCorrelationCoefficient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.true_positives.assign(np.zeros(self.num_classes), np.float32)
        self.false_positives.assign(np.zeros(self.num_classes), np.float32)
        self.false_negatives.assign(np.zeros(self.num_classes), np.float32)
        self.true_negatives.assign(np.zeros(self.num_classes), np.float32)
