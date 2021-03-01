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

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
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
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.

    Usage:

    >>> y_true = np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float32)
    >>> y_pred = np.array([[1.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    >>> metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    array([-0.33333334], dtype=float32)
    """

    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        name: str = "MatthewsCorrelationCoefficient",
        dtype: AcceptableDTypes = None,
        **kwargs,
    ):
        """Creates a Matthews Correlation Coefficient instance."""
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.true_sum = self.add_weight(
            "true_sum", shape=[self.num_classes], initializer="zeros", dtype=self.dtype
        )
        self.pred_sum = self.add_weight(
            "pred_sum", shape=[self.num_classes], initializer="zeros", dtype=self.dtype
        )
        self.num_correct = self.add_weight(
            "num_correct",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.num_samples = self.add_weight(
            "num_samples",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )

    # TODO: sample_weights
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        cov_matrix = tf.math.confusion_matrix(
            labels=tf.argmax(y_true, 1),
            predictions=tf.argmax(y_pred, 1),
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=self.dtype,
        )

        self.true_sum.assign_add(tf.reduce_sum(cov_matrix, axis=1))
        self.pred_sum = tf.reduce_sum(cov_matrix, axis=0)

        self.num_correct = tf.linalg.trace(cov_matrix)
        self.num_samples = tf.reduce_sum(self.pred_sum)

    def result(self):
        # covariance true-pred
        cov_ytyp = self.num_correct * self.num_samples - tf.tensordot(
            self.true_sum, self.pred_sum, axes=1
        )
        # covariance pred-pred
        cov_ypyp = self.num_samples ** 2 - tf.tensordot(
            self.pred_sum, self.pred_sum, axes=1
        )
        # covariance true-true
        cov_ytyt = self.num_samples ** 2 - tf.tensordot(
            self.true_sum, self.true_sum, axes=1
        )

        mcc = cov_ytyp / tf.math.sqrt(cov_ytyt * cov_ypyp)

        if tf.math.is_nan(mcc):
            mcc = tf.constant(0, dtype=self.dtype)

        return mcc

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_states(self):
        """Resets all of the metric state variables."""
        reset_value = np.zeros(self.num_classes, dtype=self.dtype)
        K.batch_set_value([(v, reset_value) for v in self.variables])
