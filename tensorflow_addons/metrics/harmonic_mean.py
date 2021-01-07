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
"""Implements HarmonicMean."""

import tensorflow as tf

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes


@tf.keras.utils.register_keras_serializable(package="Addons")
class HarmonicMean(tf.keras.metrics.Mean):
    """Compute Harmonic Mean
    The harmonic mean is a kind of mean. It can be expressed as the reciprocal of
    the arithmetic mean of the reciprocals of the given set of numbers.
    Note: `tfa.metrics.HarmonicMean` can be used the same as `tf.keras.metrics.Mean`.
    Args:
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.
    Usage:
    >>> metric = tfa.metrics.HarmonicMean()
    >>> metric.update_state([1, 4, 4])
    >>> metric.result().numpy()
    2.0
    """

    @typechecked
    def __init__(
        self, name: str = "harmonic_mean", dtype: AcceptableDTypes = None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

    def update_state(self, values, sample_weight=None) -> None:
        values = tf.cast(values, dtype=self.dtype)
        super().update_state(tf.math.reciprocal(values), sample_weight)

    def result(self) -> tf.Tensor:
        return tf.math.reciprocal_no_nan(super().result())
