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


class FBetaScore(Metric):
    """Computes F-Beta Score.

    This is the weighted harmonic mean of precision and recall.
    Output range is [0, 1].

    F-Beta = (1 + beta^2) * ((precision * recall) /
                             ((beta^2 * precision) + recall))

    `beta` parameter determines the weight given to the
    precision and recall.

    `beta < 1` gives more weight to the precision.
    `beta > 1` gives more weight to the recall.
    `beta == 1` gives equal weight to precision and recall.

    Args:
       num_classes : Number of unique classes in the dataset.
       average : Type of averaging to be performed on data.
                   Acceptable values are None, micro, macro and
                   weighted.
       beta : float. Determines the weight of precision and recall
                in harmonic mean. Acceptable values are either a number
                of float data type greater than 0.0 or a scale tensor
                of dtype tf.float32.

    Returns:
       F Beta Score: float

    Raises:
       ValueError: If the `average` has values other than
       [None, micro, macro, weighted].

       ValueError: If the `beta` value is less than or equal
       to 0.

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
    predis = tf.constant([[1, 0, 0],[1, 0, 1]],
             dtype=tf.int32)
    # F-Beta Micro
    fb_score = tfa.metrics.FBetaScore(num_classes=3,
                beta=0.4, average='micro')
    fb_score.update_state(actuals, preds)
    print('F1-Beta Score is: ',
           fb_score.result().numpy()) # 0.6666666
    # F-Beta Macro
    fb_score = tfa.metrics.FBetaScore(num_classes=3,
           beta=0.4, average='macro')
    fb_score.update_state(actuals, preds)
    print('F1-Beta Score is: ',
          fb_score.result().numpy()) # 0.33333334
    # F-Beta Weighted
    fb_score = tfa.metrics.FBetaScore(num_classes=3,
               beta=0.4, average='weighted')
    fb_score.update_state(actuals, preds)
    print('F1-Beta Score is: ',
          fb_score.result().numpy()) # 0.6666667
    # F-Beta score for each class (average=None).
    fb_score = tfa.metrics.FBetaScore(num_classes=3,
               beta=0.4, average=None)
    fb_score.update_state(actuals, preds)
    print('F1-Beta Score is: ',
         fb_score.result().numpy()) # [1. 0. 0.]
    ```
    """

    def __init__(self,
                 num_classes,
                 average=None,
                 beta=1.0,
                 name='fbeta_score',
                 dtype=tf.float32):
        super(FBetaScore, self).__init__(name=name)
        self.num_classes = num_classes
        # type check
        if not isinstance(beta, float) and beta.dtype != tf.float32:
            raise TypeError("The value of beta should be float")
        # value check
        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")
        else:
            self.beta = beta
        if average not in (None, 'micro', 'macro', 'weighted'):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [None, micro, macro, weighted]")
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
                dtype=self.dtype)
            self.false_positives = self.add_weight(
                'false_positives',
                shape=[],
                initializer='zeros',
                dtype=self.dtype)
            self.false_negatives = self.add_weight(
                'false_negatives',
                shape=[],
                initializer='zeros',
                dtype=self.dtype)
        else:
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
            self.weights_intermediate = self.add_weight(
                'weights',
                shape=[self.num_classes],
                initializer='zeros',
                dtype=self.dtype)

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # true positive
        self.true_positives.assign_add(
            tf.cast(
                tf.math.count_nonzero(y_pred * y_true, axis=self.axis),
                self.dtype))
        # false positive
        self.false_positives.assign_add(
            tf.cast(
                tf.math.count_nonzero(y_pred * (y_true - 1), axis=self.axis),
                self.dtype))
        # false negative
        self.false_negatives.assign_add(
            tf.cast(
                tf.math.count_nonzero((y_pred - 1) * y_true, axis=self.axis),
                self.dtype))
        if self.average == 'weighted':
            # variable to hold intermediate weights
            self.weights_intermediate.assign_add(
                tf.cast(tf.reduce_sum(y_true, axis=self.axis), self.dtype))

    def result(self):
        p_sum = tf.cast(self.true_positives + self.false_positives, self.dtype)
        # calculate precision
        precision = tf.math.divide_no_nan(self.true_positives, p_sum)

        r_sum = tf.cast(self.true_positives + self.false_negatives, self.dtype)
        # calculate recall
        recall = tf.math.divide_no_nan(self.true_positives, r_sum)
        # intermediate calculations
        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        f1_int = (1 + tf.math.square(self.beta)) * (tf.math.divide_no_nan(
            mul_value, add_value))
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
            "beta": self.beta,
        }
        base_config = super(FBetaScore, self).get_config()
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


class F1Score(FBetaScore):
    """Computes F1 micro, macro or weighted based on the user's choice.

    F1 score is the weighted average of precision and
    recall. Output range is [0, 1]. This works for both
    multi-class and multi-label classification.

    F-1 = (2) * ((precision * recall) / (precision + recall))

    Args:
       num_classes : Number of unique classes in the dataset.
       average : Type of averaging to be performed on data.
                 Acceptable values are `None`, `micro`, `macro` and
                 `weighted`.
                 Default value is None.
       beta : float
              Determines the weight of precision and recall in harmonic
              mean. It's value is 1.0 for F1 score.

    Returns:
       F1 Score: float

    Raises:
       ValueError: If the `average` has values other than
       [None, micro, macro, weighted].

       ValueError: If the `beta` value is less than or equal
       to 0.

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
    # F1 Micro
    output = tfa.metrics.F1Score(num_classes=3,
              average='micro')
    output.update_state(actuals, preds)
    print('F1 Micro score is: ',
            output.result().numpy()) # 0.6666667
    # F1 Macro
    output = tfa.metrics.F1Score(num_classes=3,
                average='macro')
    output.update_state(actuals, preds)
    print('F1 Macro score is: ',
            output.result().numpy()) # 0.33333334
    # F1 weighted
    output = tfa.metrics.F1Score(num_classes=3,
              average='weighted')
    output.update_state(actuals, preds)
    print('F1 Weighted score is: ',
            output.result().numpy()) # 0.6666667
    # F1 score for each class (average=None).
    output = tfa.metrics.F1Score(num_classes=3)
    output.update_state(actuals, preds)
    print('F1 score is: ',
            output.result().numpy()) # [1. 0. 0.]
    ```
    """

    def __init__(self, num_classes, average, name='f1_score',
                 dtype=tf.float32):
        super(F1Score, self).__init__(
            num_classes, average, 1.0, name=name, dtype=dtype)

    def get_config(self):
        base_config = super(F1Score, self).get_config()
        del base_config["beta"]
        return base_config
