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
"""Implements Multi-label confusion matrix scores."""

import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
import numpy as np

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike


class MultiLabelConfusionMatrix(Metric):
    """Computes Multi-label confusion matrix.

    Class-wise confusion matrix is computed for the
    evaluation of classification.

    If multi-class input is provided, it will be treated
    as multilabel data.

    Consider classification problem with two classes
    (i.e num_classes=2).

    Resultant matrix `M` will be in the shape of `(num_classes, 2, 2)`.

    Every class `i` has a dedicated matrix of shape `(2, 2)` that contains:

    - true negatives for class `i` in `M(0,0)`
    - false positives for class `i` in `M(0,1)`
    - false negatives for class `i` in `M(1,0)`
    - true positives for class `i` in `M(1,1)`

    Args:
        num_classes: `int`, the number of labels the prediction task can have.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Usage:

    >>> # multilabel confusion matrix
    >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
    >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
    >>> metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=3)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()  #doctest: -DONT_ACCEPT_BLANKLINE
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[0., 1.],
            [1., 0.]]], dtype=float32)
    >>> # if multiclass input is provided
    >>> y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int32)
    >>> y_pred = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.int32)
    >>> metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=3)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy() #doctest: -DONT_ACCEPT_BLANKLINE
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [1., 0.]],
    <BLANKLINE>
           [[1., 1.],
            [0., 0.]]], dtype=float32)

    """

    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        name: str = "Multilabel_confusion_matrix",
        dtype: AcceptableDTypes = None,
        **kwargs
    ):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            "true_positives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.false_negatives = self.add_weight(
            "false_negatives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.true_negatives = self.add_weight(
            "true_negatives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            warnings.warn(
                "`sample_weight` is not None. Be aware that MultiLabelConfusionMatrix "
                "does not take `sample_weight` into account when computing the metric "
                "value."
            )

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        # true positive
        true_positive = tf.math.count_nonzero(y_true * y_pred, 0)
        # predictions sum
        pred_sum = tf.math.count_nonzero(y_pred, 0)
        # true labels sum
        true_sum = tf.math.count_nonzero(y_true, 0)
        false_positive = pred_sum - true_positive
        false_negative = true_sum - true_positive
        y_true_negative = tf.math.not_equal(y_true, 1)
        y_pred_negative = tf.math.not_equal(y_pred, 1)
        true_negative = tf.math.count_nonzero(
            tf.math.logical_and(y_true_negative, y_pred_negative), axis=0
        )

        # true positive state update
        self.true_positives.assign_add(tf.cast(true_positive, self.dtype))
        # false positive state update
        self.false_positives.assign_add(tf.cast(false_positive, self.dtype))
        # false negative state update
        self.false_negatives.assign_add(tf.cast(false_negative, self.dtype))
        # true negative state update
        self.true_negatives.assign_add(tf.cast(true_negative, self.dtype))

    def result(self):
        flat_confusion_matrix = tf.convert_to_tensor(
            [
                self.true_negatives,
                self.false_positives,
                self.false_negatives,
                self.true_positives,
            ]
        )
        # reshape into 2*2 matrix
        confusion_matrix = tf.reshape(tf.transpose(flat_confusion_matrix), [-1, 2, 2])

        return confusion_matrix

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_states(self):
        reset_value = np.zeros(self.num_classes, dtype=np.int32)
        K.batch_set_value([(v, reset_value) for v in self.variables])
