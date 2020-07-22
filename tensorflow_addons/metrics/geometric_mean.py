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
"""Implements GeometricMean."""

import warnings

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes


@tf.keras.utils.register_keras_serializable(package="Addons")
class GeometricMean(Metric):
    """Compute Geometric Mean

    The geometric mean is a kind of mean. Unlike the arithmetic mean
    that uses the sum of values, it uses the product of the values to
    represent typical values in a set of numbers.

    Note: `tfa.metrics.GeometryMean` can be used the same as `tf.keras.metrics.Mean`

    Usage:
    ```python
    >>> m = tfa.metrics.GeometricMean()
    >>> m.update_state([1, 3, 5, 7, 9])
    >>> m.result().numpy()
    3.9362833
    ```
    """

    @typechecked
    def __init__(
        self, name: str = "geometric_mean", dtype: AcceptableDTypes = None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.accure = self.add_weight(
            "accure", shape=None, initializer="zeros", dtype=dtype
        )
        self.count = self.add_weight(
            "count", shape=None, initializer="zeros", dtype=dtype
        )

    def update_state(self, y_true, y_pred=None, sample_weight=None) -> None:
        if y_pred is not None:
            warnings.warn("`y_pred` is not None. `y_pred` is not used in GeometricMean")
        if sample_weight is not None:
            warnings.warn(
                "`sample_weight` is not None. Be aware that GeometricMean"
                "does not take `sample_weight` into account when computing"
                " the metric value."
            )
        values = y_true
        if not isinstance(values, tf.Tensor):
            values = tf.convert_to_tensor(values, dtype=self.dtype)
        elif values.dtype != self.dtype:
            values = tf.cast(values, dtype=self.dtype)

        self.count.assign_add(np.size(values))
        if not tf.math.is_inf(self.accure):
            log_v = tf.math.log(values)
            if log_v.shape != []:
                log_v = tf.reduce_sum(log_v)
            self.accure.assign_add(log_v)

    def result(self) -> tf.Tensor:
        if tf.math.is_inf(self.accure):
            return tf.constant(0, dtype=self.dtype)
        ret = tf.math.exp(self.accure / self.count)
        print(self.accure, self.count)
        if ret.dtype is not self.dtype:
            return tf.cast(ret, dtype=self.dtype)
        return ret

    def reset_states(self) -> None:
        K.batch_set_value([(v, 0) for v in self.variables])
