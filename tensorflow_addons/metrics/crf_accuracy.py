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
"""Implements Accuracy for Conditional Random Field."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops.losses import util as tf_losses_utils

from tensorflow_addons.utils import keras_utils


def _crf_accuracy(y_true, y_pred):
    crf_layer = y_pred._keras_history[0]
    return crf_layer.get_accuracy(y_true, y_pred)


@keras_utils.register_keras_custom_object
class ConditionalRandomFieldAccuracy(tf.keras.metrics.Mean):
    """Wraps a stateless metric function with the Mean metric."""

    def __init__(self, name='crf_accuracy', dtype=None):
        """Creates a `MeanMetricWrapper` instance.

        Args:
          fn: The metric function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super(ConditionalRandomFieldAccuracy, self).__init__(
            name=name, dtype=dtype)

        self._fn = _crf_accuracy

    def __new__(cls, *args, **kwargs):
        obj = Layer.__new__(cls)

        # A hack here, origianl base class (tf.keras.metrics.Metric)
        # will convert update_state using tf.function
        # but which will cause problem related to _keras_history
        update_state_fn = obj.update_state

        obj.update_state = types.MethodType(
            metrics_utils.update_state_wrapper(update_state_fn), obj)
        obj.result = types.MethodType(
            metrics_utils.result_wrapper(obj.result), obj)
        return obj

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.dtypes.cast(y_true, self._dtype)

        # cast operation will drop _keras_history info, which is vital to this metrics
        # so, store it and then restore it later
        y_pred_keras_history = y_pred._keras_history
        y_pred = tf.dtypes.cast(y_pred, self._dtype)
        y_pred._keras_history = y_pred_keras_history

        [y_true, y_pred], sample_weight = \
            metrics_utils.ragged_assert_compatible_and_get_flat_values(
                [y_true, y_pred], sample_weight)
        y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true)

        matches = self._fn(y_true, y_pred)
        return super(ConditionalRandomFieldAccuracy, self).update_state(
            matches, sample_weight=sample_weight)

    def get_config(self):
        config = {}
        base_config = super(ConditionalRandomFieldAccuracy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
