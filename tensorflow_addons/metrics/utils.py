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
"""Utilities for metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf


class MeanMetricWrapper(tf.keras.metrics.Mean):
    """Wraps a stateless metric function with the Mean metric."""

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        """Creates a `MeanMetricWrapper` instance.
        Args:
          fn: The metric function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        `y_true` and `y_pred` should have the same shape.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1.
            Can be a `Tensor` whose rank is either 0, or the same rank as
            `y_true`, and must be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        # TODO: Add checks for ragged tensors and dimensions:
        #   `ragged_assert_compatible_and_get_flat_values`
        #   and `squeeze_or_expand_dimensions`
        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super(MeanMetricWrapper, self).update_state(
            matches, sample_weight=sample_weight)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = v
        base_config = super(MeanMetricWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
