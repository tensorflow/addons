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
"""Implements F1 scores."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.metrics import Metric
import numpy as np


class F1Score(Metric):
    """Calculates F1 micro, macro or weighted based on the user's choice.

    F1 score is the weighted average of precision and
    recall. Output range is [0, 1]. This works for both
    multi-class and multi-label classification.


    Args:
       num_classes : Number of unique classes in the dataset.
       average : Type of averaging to be performed on data.
                 Acceptable values are None, micro, macro and
                 weighted.
                 Default value is None.

    Returns:
       F1 score: float

    Raises:
       ValueError: If the `average` has values other than
       [None, micro, macro. weighted].

    `average` parameter behavior:

    1. If `None` is specified as an input, scores for each
       class are returned.

    2. If `micro` is specified, metrics like true positivies,
       false positives and false negatives are computed
       globally.

    3. If `macro` is specified, metrics like true positivies,
       false positives and false negatives are computed for
       each class and their unweighted mean is returned.
       Imbalance in dataset is not taken into account for
       calculating the score

    4. If `weighted` is specified, metrics are computed for
       each class and returns the mean weighted by the
       number of true instances in each class taking data
       imbalance into account.

    Usage:
    ```python
    actuals = tf.constant([[1, 1, 0],[1, 0, 0]],
              dtype=tf.int32)
    preds = tf.constant([[1, 0, 0],[1, 0, 1]],
            dtype=tf.int32)
    output = tf.keras.metrics.F1Score(num_classes=3,
              average='micro')
    output.update_state(actuals, predictions)
    print('F1 Micro score is: ',
            output.result().numpy()) # 0.6666667
    ```
    """

    def __init__(self,
                 num_classes,
                 average=None,
                 name='f1_score',
                 dtype=tf.float32):
        super(F1Score, self).__init__(name=name)
        self.num_classes = num_classes
        if average not in (None, 'micro', 'macro', 'weighted'):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [micro, macro, weighted]")
        else:
            self.average = average
            if self.average == 'micro':
                self.axis = None
            else:
                self.axis = 0
        if self.average == 'micro':
            self.true_positives = self.add_weight(
                'true_positives',
                shape=[],
                initializer='zeros',
                dtype=tf.float32)
            self.false_positives = self.add_weight(
                'false_positives',
                shape=[],
                initializer='zeros',
                dtype=tf.float32)
            self.false_negatives = self.add_weight(
                'false_negatives',
                shape=[],
                initializer='zeros',
                dtype=tf.float32)
        else:
            self.true_positives = self.add_weight(
                'true_positives',
                shape=[self.num_classes],
                initializer='zeros',
                dtype=tf.float32)
            self.false_positives = self.add_weight(
                'false_positives',
                shape=[self.num_classes],
                initializer='zeros',
                dtype=tf.float32)
            self.false_negatives = self.add_weight(
                'false_negatives',
                shape=[self.num_classes],
                initializer='zeros',
                dtype=tf.float32)
            self.weights_intermediate = self.add_weight(
                'weights',
                shape=[self.num_classes],
                initializer='zeros',
                dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # true positive
        self.true_positives.assign_add(
            tf.cast(
                tf.math.count_nonzero(y_pred * y_true, axis=self.axis),
                tf.float32))
        # false positive
        self.false_positives.assign_add(
            tf.cast(
                tf.math.count_nonzero(y_pred * (y_true - 1), axis=self.axis),
                tf.float32))
        # false negative
        self.false_negatives.assign_add(
            tf.cast(
                tf.math.count_nonzero((y_pred - 1) * y_true, axis=self.axis),
                tf.float32))
        if self.average == 'weighted':
            # variable to hold intermediate weights
            self.weights_intermediate.assign_add(
                tf.cast(tf.reduce_sum(y_true, axis=self.axis), tf.float32))

    def result(self):
        p_sum = tf.cast(self.true_positives + self.false_positives, tf.float32)
        # calculate precision
        precision = tf.math.divide_no_nan(self.true_positives, p_sum)

        r_sum = tf.cast(self.true_positives + self.false_negatives, tf.float32)
        # calculate recall
        recall = tf.math.divide_no_nan(self.true_positives, r_sum)

        mul_value = 2 * precision * recall
        add_value = precision + recall
        f1_int = tf.math.divide_no_nan(mul_value, add_value)
        # f1 score
        if self.average is not None:
            f1_score = tf.reduce_mean(f1_int)
        else:
            f1_score = f1_int
        # condition for weighted f1 score
        if self.average == 'weighted':
            f1_int_weights = tf.math.divide_no_nan(
                self.weights_intermediate,
                tf.reduce_sum(self.weights_intermediate))
            # weighted f1 score calculation
            f1_score = tf.reduce_sum(f1_int * f1_int_weights)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
        }
        base_config = super(F1Score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        # reset state of the variables to zero
        if self.average == 'micro':
            self.true_positives.assign(0)
            self.false_positives.assign(0)
            self.false_negatives.assign(0)
        else:
            self.true_positives.assign(np.zeros(self.num_classes), np.float32)
            self.false_positives.assign(np.zeros(self.num_classes), np.float32)
            self.false_negatives.assign(np.zeros(self.num_classes), np.float32)
            self.weights_intermediate.assign(
                np.zeros(self.num_classes), np.float32)
