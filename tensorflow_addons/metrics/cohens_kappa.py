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
"""Implements Cohen's Kappa."""

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike

from typeguard import typechecked
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
class CohenKappa(Metric):
    """Computes Kappa score between two raters.

    The score lies in the range `[-1, 1]`. A score of -1 represents
    complete disagreement between two raters whereas a score of 1
    represents complete agreement between the two raters.
    A score of 0 means agreement by chance.

    Note: As of now, this implementation considers all labels
    while calculating the Cohen's Kappa score.

    Args:
        num_classes: Number of unique classes in your dataset.
        weightage: (optional) Weighting to be considered for calculating
            kappa statistics. A valid value is one of
            [None, 'linear', 'quadratic']. Defaults to `None`
        sparse_labels: (bool) Valid only for multi-class scenario.
            If True, ground truth labels are expected to be integers
            and not one-hot encoded.
        regression: (bool) If set, that means the problem is being treated
            as a regression problem where you are regressing the predictions.
            **Note:** If you are regressing for the values, the the output layer
            should contain a single unit.
        name: (optional) String name of the metric instance
        dtype: (optional) Data type of the metric result. Defaults to `None`.

    Raises:
        ValueError: If the value passed for `weightage` is invalid
        i.e. not any one of [None, 'linear', 'quadratic'].

    Usage:

    >>> y_true = np.array([4, 4, 3, 4, 2, 4, 1, 1], dtype=np.int32)
    >>> y_pred = np.array([4, 4, 3, 4, 4, 2, 1, 1], dtype=np.int32)
    >>> weights = np.array([1, 1, 2, 5, 10, 2, 3, 3], dtype=np.int32)
    >>> metric = tfa.metrics.CohenKappa(num_classes=5, sparse_labels=True)
    >>> metric.update_state(y_true , y_pred)
    <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
     array([[0., 0., 0., 0., 0.],
            [0., 2., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 3.]], dtype=float32)>
    >>> result = metric.result()
    >>> result.numpy()
    0.61904764
    >>> # To use this with weights, sample_weight argument can be used.
    >>> metric = tfa.metrics.CohenKappa(num_classes=5, sparse_labels=True)
    >>> metric.update_state(y_true , y_pred , sample_weight=weights)
    <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
     array([[ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  6.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0., 10.],
            [ 0.,  0.,  0.,  2.,  0.],
            [ 0.,  0.,  2.,  0.,  7.]], dtype=float32)>
    >>> result = metric.result()
    >>> result.numpy()
     0.37209308

    Usage with `tf.keras` API:

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> x = tf.keras.layers.Dense(10)(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    >>> model.compile('sgd', loss='mse', metrics=[tfa.metrics.CohenKappa(num_classes=3, sparse_labels=True)])
    """

    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        name: str = "cohen_kappa",
        weightage: Optional[str] = None,
        sparse_labels: bool = False,
        regression: bool = False,
        dtype: AcceptableDTypes = None,
    ):
        """Creates a `CohenKappa` instance."""
        super().__init__(name=name, dtype=dtype)

        if weightage not in (None, "linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")

        if num_classes == 2:
            self._update = self._update_binary_class_model
        elif num_classes > 2:
            self._update = self._update_multi_class_model
        else:
            raise ValueError(
                """Number of classes must be
                              greater than or euqal to two"""
            )

        self.weightage = weightage
        self.num_classes = num_classes
        self.regression = regression
        self.sparse_labels = sparse_labels
        self.conf_mtx = self.add_weight(
            "conf_mtx",
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.float32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix condition statistics.

        Args:
          y_true: Labels assigned by the first annotator with shape
            `[num_samples,]`.
          y_pred: Labels assigned by the second annotator with shape
            `[num_samples,]`. The kappa statistic is symmetric,
            so swapping `y_true` and `y_pred` doesn't change the value.
          sample_weight (optional): for weighting labels in confusion matrix
            Defaults to `None`. The dtype for weights should be the same
            as the dtype for confusion matrix. For more details,
            please check `tf.math.confusion_matrix`.

        Returns:
          Update op.
        """
        return self._update(y_true, y_pred, sample_weight)

    def _update_binary_class_model(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.int64)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_pred = tf.cast(y_pred > 0.5, dtype=tf.int64)
        return self._update_confusion_matrix(y_true, y_pred, sample_weight)

    @tf.function
    def _update_multi_class_model(self, y_true, y_pred, sample_weight=None):
        v = tf.argmax(y_true, axis=1) if not self.sparse_labels else y_true
        y_true = tf.cast(v, dtype=tf.int64)

        y_pred = self._cast_ypred(y_pred)

        return self._update_confusion_matrix(y_true, y_pred, sample_weight)

    @tf.function
    def _cast_ypred(self, y_pred):
        if tf.rank(y_pred) > 1:
            if not self.regression:
                y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int64)
            else:
                y_pred = tf.math.round(tf.math.abs(y_pred))
                y_pred = tf.cast(y_pred, dtype=tf.int64)
        else:
            y_pred = tf.cast(y_pred, dtype=tf.int64)
        return y_pred

    @tf.function
    def _safe_squeeze(self, y):
        y = tf.squeeze(y)

        # Check for scalar result
        if tf.rank(y) == 0:
            y = tf.expand_dims(y, 0)

        return y

    def _update_confusion_matrix(self, y_true, y_pred, sample_weight):
        y_true = self._safe_squeeze(y_true)
        y_pred = self._safe_squeeze(y_pred)

        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=tf.float32,
        )

        return self.conf_mtx.assign_add(new_conf_mtx)

    def result(self):
        nb_ratings = tf.shape(self.conf_mtx)[0]
        weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.float32)

        # 2. Create a weight matrix
        if self.weightage is None:
            diagonal = tf.zeros([nb_ratings], dtype=tf.float32)
            weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
        else:
            weight_mtx += tf.cast(tf.range(nb_ratings), dtype=tf.float32)
            weight_mtx = tf.cast(weight_mtx, dtype=self.dtype)

            if self.weightage == "linear":
                weight_mtx = tf.abs(weight_mtx - tf.transpose(weight_mtx))
            else:
                weight_mtx = tf.pow((weight_mtx - tf.transpose(weight_mtx)), 2)

        weight_mtx = tf.cast(weight_mtx, dtype=self.dtype)

        # 3. Get counts
        actual_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=1)
        pred_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=0)

        # 4. Get the outer product
        out_prod = pred_ratings_hist[..., None] * actual_ratings_hist[None, ...]

        # 5. Normalize the confusion matrix and outer product
        conf_mtx = self.conf_mtx / tf.reduce_sum(self.conf_mtx)
        out_prod = out_prod / tf.reduce_sum(out_prod)

        conf_mtx = tf.cast(conf_mtx, dtype=self.dtype)
        out_prod = tf.cast(out_prod, dtype=self.dtype)

        # 6. Calculate Kappa score
        numerator = tf.reduce_sum(conf_mtx * weight_mtx)
        denominator = tf.reduce_sum(out_prod * weight_mtx)
        return tf.cond(
            tf.math.is_nan(denominator),
            true_fn=lambda: 0.0,
            false_fn=lambda: 1 - (numerator / denominator),
        )

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
            "sparse_labels": self.sparse_labels,
            "regression": self.regression,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        """Resets all of the metric state variables."""

        for v in self.variables:
            K.set_value(
                v,
                np.zeros((self.num_classes, self.num_classes), v.dtype.as_numpy_dtype),
            )

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()
