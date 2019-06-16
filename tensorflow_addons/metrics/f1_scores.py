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
    """
    Calculates F1 micro, macro or weighted based on the
    user's choice
    """

    def __init__(self, num_classes, average=None,
                 name='f1_score', dtype=tf.float32):
        super(F1Score, self).__init__(name=name)
        self.num_classes = num_classes
        if average not in (None, 'micro', 'macro', 'weighted'):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [micro, macro, weighted]")
        else:
            self.average = average
            if self.average == 'micro':
                self.axis = None
            if self.average == 'macro' or self.average == 'weighted':
                self.axis = 0
        if self.average == 'micro':
            self.true_positives = self.add_weight('true_positives',
                                                  shape=[],
                                                  initializer='zeros',
                                                  dtype=tf.float32)
            self.false_positives = self.add_weight('false_positives',
                                                   shape=[],
                                                   initializer='zeros',
                                                   dtype=tf.float32)
            self.false_negatives = self.add_weight('false_negatives',
                                                   shape=[],
                                                   initializer='zeros',
                                                   dtype=tf.float32)
        else:
            self.true_positives = self.add_weight('true_positives',
                                                  shape=[self.num_classes],
                                                  initializer='zeros',
                                                  dtype=tf.float32)
            self.false_positives = self.add_weight('false_positives',
                                                   shape=[self.num_classes],
                                                   initializer='zeros',
                                                   dtype=tf.float32)
            self.false_negatives = self.add_weight('false_negatives',
                                                   shape=[self.num_classes],
                                                   initializer='zeros',
                                                   dtype=tf.float32)
            self.weights_intermediate = self.add_weight('weights', shape=[
                self.num_classes], initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # true positive
        self.true_positives.assign_add(tf.cast(
            tf.math.count_nonzero(y_pred * y_true, axis=self.axis),
            tf.float32))
        # false positive
        self.false_positives.assign_add(
            tf.cast(tf.math.count_nonzero(y_pred * (y_true - 1),
                                          axis=self.axis), tf.float32))
        # false negative
        self.false_negatives.assign_add(tf.cast(tf.math.count_nonzero(
            (y_pred - 1) * y_true, axis=self.axis), tf.float32))
        if self.average == 'weighted':
            # variable to hold intermediate weights
            self.weights_intermediate.assign_add(tf.cast(
                tf.reduce_sum(y_true, axis=self.axis), tf.float32))

    def result(self):
        p_sum = tf.cast(self.true_positives + self.false_positives,
                        tf.float32)
        # calculate precision
        precision_macro = tf.math.divide_no_nan(self.true_positives,
                                                p_sum)

        r_sum = tf.cast(self.true_positives + self.false_negatives,
                        tf.float32)
        # calculate recall
        recall_macro = tf.math.divide_no_nan(self.true_positives,
                                             r_sum)

        mul_value = 2 * precision_macro * recall_macro
        add_value = precision_macro + recall_macro
        f1_macro_int = tf.math.divide_no_nan(mul_value, add_value)
        # f1 macro score
        f1_score = tf.reduce_mean(f1_macro_int)
        # condition for weighted f1 score
        if self.average == 'weighted':
            f1_int_weights = tf.cast(tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(
                    self.weights_intermediate)),
                tf.float32)
            # weighted f1 score calculation
            f1_score = tf.reduce_sum(f1_macro_int * f1_int_weights)

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
            self.weights_intermediate.assign(np.zeros(self.num_classes),
                                             np.float32)
