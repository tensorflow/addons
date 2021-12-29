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

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils.types import AcceptableDTypes

from typeguard import typechecked
from typing import Optional, Callable


class MeanMetricWrapper(tf.keras.metrics.Mean):
    """Wraps a stateless metric function with the Mean metric."""

    @typechecked
    def __init__(
        self,
        fn: Callable,
        name: Optional[str] = None,
        dtype: AcceptableDTypes = None,
        **kwargs,
    ):
        """Creates a `MeanMetricWrapper` instance.
        Args:
          fn: The metric function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(name=name, dtype=dtype)
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
        return super().update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {k: v for k, v in self._fn_kwargs.items()}
        base_config = super().get_config()
        return {**base_config, **config}


def _get_model(metric, num_output):
    # Test API comptibility with tf.keras Model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(num_output, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["acc", metric]
    )

    data = np.random.random((10, 3))
    labels = np.random.random((10, num_output))
    model.fit(data, labels, epochs=1, batch_size=5, verbose=0)


def sample_weight_shape_match(v, sample_weight):
    if sample_weight is None:
        return tf.ones_like(v)
    if np.size(sample_weight) == 1:
        return tf.fill(v.shape, sample_weight)
    return tf.convert_to_tensor(sample_weight)
