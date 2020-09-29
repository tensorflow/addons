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

import tensorflow as tf
from tensorflow_addons.metrics.utils import MeanMetricWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike, AcceptableDTypes

from typeguard import typechecked
from typing import Union, Optional


def hamming_distance(actuals: TensorLike, predictions: TensorLike) -> tf.Tensor:
    """Computes hamming distance.

    Hamming distance is for comparing two binary strings.
    It is the number of bit positions in which two bits
    are different.

    Args:
        actuals: actual target value.
        predictions: predicted value.

    Returns:
        hamming distance: float.

    Usage:

    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=np.int32)
    >>> y_pred = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1], dtype=np.int32)
    >>> hamming_distance(y_true, y_pred).numpy()
    0.3

    """
    result = tf.not_equal(actuals, predictions)
    not_eq = tf.reduce_sum(tf.cast(result, tf.float32))
    ham_distance = tf.math.divide_no_nan(not_eq, len(result))
    return ham_distance


def hamming_loss_fn(
    y_true: TensorLike,
    y_pred: TensorLike,
    threshold: Union[FloatTensorLike, None],
    mode: str,
) -> tf.Tensor:
    """Computes hamming loss.

    Hamming loss is the fraction of wrong labels to the total number
    of labels.

    In multi-class classification, hamming loss is calculated as the
    hamming distance between `y_true` and `y_pred`.
    In multi-label classification, hamming loss penalizes only the
    individual labels.

    Args:
        y_true: actual target value.
        y_pred: predicted target value.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        mode: multi-class or multi-label.

    Returns:
        hamming loss: float.
    """
    if mode not in ["multiclass", "multilabel"]:
        raise TypeError("mode must be either multiclass or multilabel]")

    if threshold is None:
        threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        # make sure [0, 0, 0] doesn't become [1, 1, 1]
        # Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
    else:
        y_pred = y_pred > threshold

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    if mode == "multiclass":
        nonzero = tf.cast(tf.math.count_nonzero(y_true * y_pred, axis=-1), tf.float32)
        return 1.0 - nonzero

    else:
        nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
        return nonzero / y_true.get_shape()[-1]


class HammingLoss(MeanMetricWrapper):
    """Computes hamming loss.

    Hamming loss is the fraction of wrong labels to the total number
    of labels.

    In multi-class classification, hamming loss is calculated as the
    hamming distance between `y_true` and `y_pred`.
    In multi-label classification, hamming loss penalizes only the
    individual labels.

    Args:
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        mode: multi-class or multi-label.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Usage:

    >>> # multi-class hamming loss
    >>> metric = HammingLoss(mode='multiclass', threshold=0.6)
    >>> y_true = np.array([[1.0, 0.0, 0.0, 0.0],
    ...                    [0.0, 0.0, 1.0, 0.0],
    ...                    [0.0, 0.0, 0.0, 1.0],
    ...                    [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    >>> y_pred = np.array([[0.8, 0.1, 0.1, 0.0],
    ...                    [0.2, 0.0, 0.8, 0.0],
    ...                    [0.05, 0.05, 0.1, 0.8],
    ...                    [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    >>> metric.update_state(y_true, y_pred)
    <tf.Variable 'UnreadVariable' shape=() dtype=float32, numpy=4.0>
    >>> metric.result().numpy()
    0.25
    >>> # multi-label hamming loss
    >>> metric = HammingLoss(mode='multilabel', threshold=0.8)
    >>> y_true = np.array([[1, 0, 1, 0],
    ...                    [0, 1, 0, 1],
    ...                    [0, 0, 0, 1]], dtype=np.int32)
    >>> y_pred = np.array([[0.82, 0.5, 0.90, 0],
    ...                    [0, 1, 0.4, 0.98],
    ...                    [0.89, 0.79, 0, 0.3]], dtype=np.float32)
    >>> metric.update_state(y_true, y_pred)
    <tf.Variable 'UnreadVariable' shape=() dtype=float32, numpy=3.0>
    >>> metric.result().numpy()
    0.16666667
    """

    @typechecked
    def __init__(
        self,
        mode: str,
        name: str = "hamming_loss",
        threshold: Optional[FloatTensorLike] = None,
        dtype: AcceptableDTypes = None,
        **kwargs
    ):
        super().__init__(
            hamming_loss_fn, name=name, dtype=dtype, mode=mode, threshold=threshold
        )
