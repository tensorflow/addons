# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Implements GeometricMean."""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes
from tensorflow_addons.metrics.utils import sample_weight_shape_match


@tf.keras.utils.register_keras_serializable(package="Addons")
class GeometricMean(Metric):
    """Compute Geometric Mean

    The geometric mean is a kind of mean. Unlike the arithmetic mean
    that uses the sum of values, it uses the product of the values to
    represent typical values in a set of numbers.

    Note: `tfa.metrics.GeometricMean` can be used the same as `tf.keras.metrics.Mean`

    Usage:

    >>> m = tfa.metrics.GeometricMean()
    >>> m.update_state([1, 3, 5, 7, 9])
    >>> m.result().numpy()
    3.9362833

    """

    @typechecked
    def __init__(
        self, name: str = "geometric_mean", dtype: AcceptableDTypes = None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(
            "total", shape=None, initializer="zeros", dtype=dtype
        )
        self.count = self.add_weight(
            "count", shape=None, initializer="zeros", dtype=dtype
        )

    def update_state(self, values, sample_weight=None) -> None:
        values = tf.cast(values, dtype=self.dtype)
        sample_weight = sample_weight_shape_match(values, sample_weight)
        sample_weight = tf.cast(sample_weight, dtype=self.dtype)

        self.count.assign_add(tf.reduce_sum(sample_weight))
        if not tf.math.is_inf(self.total):
            log_v = tf.math.log(values)
            log_v = tf.math.multiply(sample_weight, log_v)
            log_v = tf.reduce_sum(log_v)
            self.total.assign_add(log_v)

    def result(self) -> tf.Tensor:
        if tf.math.is_inf(self.total):
            return tf.constant(0, dtype=self.dtype)
        ret = tf.math.exp(self.total / self.count)
        return tf.cast(ret, dtype=self.dtype)

    def reset_states(self) -> None:
        K.batch_set_value([(v, 0) for v in self.variables])
