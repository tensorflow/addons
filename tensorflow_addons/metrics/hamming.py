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
"""Implements Hamming distance and loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics.utils import MeanMetricWrapper


def hamming_distance(y_true, y_pred):
    """Computes hamming distance.

    Hamming distance is for comparing two binary strings.
    It is the number of bit positions in which two bits
    are different.

    :param y_true: actual target value
    :param y_pred: predicted value
    :return: hamming distance

    ```python
    actuals = tf.constant([1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
                          dtype=tf.int32)
    predictions = tf.constant([1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                              dtype=tf.int32)
    result = hamming_distance(actuals, predictions)
    print('Hamming distance: ', result.numpy())
    ```
    """
    result = tf.not_equal(actuals, predictions)
    not_eq = tf.reduce_sum(tf.cast(result, tf.float32))
    ham_distance = tf.math.divide_no_nan(not_eq, len(result))
    return ham_distance


def hamming_loss_fn(y_true, y_pred, mode):
    """Computes hamming loss.

    Hamming loss is the fraction of wrong labels to the total number
    of labels.

    In multi-class classification, hamming loss is calculated as the
    hamming distance between `actual` and `predictions`.
    In multi-label classification, hamming loss penalizes only the
    individual labels.

    :param y_true: actual target value
    :param y_pred: predicted target value
    :param mode: multi-class or multi-label
    :return: hamming loss

    ```python
    # multi-class hamming loss
    hl = HammingLoss(mode='multiclass')
    actuals = tf.constant([[1, 0, 0, 0],[0, 0, 1, 0],
                       [0, 0, 0, 1],[0, 1, 0, 0]],
                      dtype=np.int32)
    predictions = tf.constant([[1, 0, 0, 0], [0, 0, 1, 0],
                           [0, 0, 0, 1], [1, 0, 0, 0]],
                          dtype=np.int32)
    hl.update_state(actuals, predictions)
    print('Hamming loss: ', hl.result().numpy()) # 0.25

    # multi-label hamming loss
    hl = HammingLoss(mode='multilabel')
    actuals = tf.constant([[1, 0, 1, 0],[0, 1, 0, 1],
                       [0, 0, 0,1]], dtype=tf.int32)
    predictions = tf.constant([[1, 0, 1, 0],[0, 1, 0, 1],
                           [1, 0, 0, 0]], dtype=tf.int32)
    hl.update_state(actuals, predictions)
    print('Hamming loss: ', hl.result().numpy()) # 0.16666667
    ```
    """
    if mode not in ['multiclass', 'multilabel']:
        raise TypeError('mode must be: [multiclass, multilabel]')

    if mode == 'multiclass':
        nonzero = tf.cast(
            tf.math.count_nonzero(y_true * y_pred, axis=-1), tf.float32)
        return 1.0 - nonzero

    else:
        nonzero = tf.cast(
            tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
        return nonzero / y_true.get_shape()[-1]


class HammingLoss(MeanMetricWrapper):
    def __init__(self, mode, name='hamming_loss', dtype=tf.float32):
        super(HammingLoss, self).__init__(
            hamming_loss_fn, name, dtype=dtype, mode=mode)
