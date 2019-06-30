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
"""Implements Multilabel confusion matrix scores."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.metrics import Metric
import numpy as np


class MultiLabelConfusionMatrix(Metric):
    def __init__(self,
                 num_classes,
                 name='Multilabel_confusion_matrix',
                 dtype=tf.float32):
        super(MultiLabelConfusionMatrix, self).__init__(name=name)
        self.num_classes = num_classes
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
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # intermediate true positive calculation
        true_positive_inter = y_true * y_pred
        # true positives
        tp = tf.math.count_nonzero(true_positive_inter, 0)
        # predictions sum
        pred_sum = tf.math.count_nonzero(y_pred, 0)
        # true labels sum
        true_sum = tf.math.count_nonzero(y_true, 0)
        # false positives
        print(tf.size(tp))
        print(tp)
        fp = pred_sum - tp
        # false negatives
        fn = true_sum - tp
        # true negatives
        print('in')
        tn = y_true.get_shape()[0] - tp - fp - fn

        # true positive state update
        self.true_positives.assign_add(
            tf.cast(tp, tf.float32))
        # false positive state update
        self.false_positives.assign_add(
            tf.cast(fp, tf.float32))
        # false negative state update
        self.false_negatives.assign_add(
            tf.cast(fn, tf.float32))
        # true negative state update
        self.true_negatives.assign_add(
            tf.cast(tn, tf.float32))

    def result(self):
        tf_int = tf.convert_to_tensor([self.true_negatives, self.false_positives,
                                       self.false_negatives,
                                       self.true_positives])
        result = tf.reshape(tf.transpose(tf_int), [-1, 2, 2])

        return result

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
        }
        base_config = super(MultiLabelConfusionMatrix, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.true_positives.assign(np.zeros(self.num_classes), np.float32)
        self.false_positives.assign(np.zeros(self.num_classes), np.float32)
        self.false_negatives.assign(np.zeros(self.num_classes), np.float32)
        self.true_negatives.assign(np.zeros(self.num_classes), np.float32)
