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

import tensorflow as tf

from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
def tanhshrink(x: types.TensorLike) -> tf.Tensor:
    r"""Tanh shrink function.

    Applies the element-wise function:

    $$
    \mathrm{tanhshrink}(x) = x - \tanh(x).
    $$

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> tfa.activations.tanhshrink(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.23840582,  0.        ,  0.23840582], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `float16`, `float32`, `float64`.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)

    return _tanhshrink_py(x)


def _tanhshrink_py(x):
    return x - tf.math.tanh(x)
