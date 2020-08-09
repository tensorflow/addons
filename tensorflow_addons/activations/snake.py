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

import tensorflow as tf

from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
def snake(x: types.TensorLike, frequency: types.Number = 1) -> tf.Tensor:
    r"""Snake activation to learn periodic functions.

    Computes snake activation:

    $$
    \mathrm{snake}(x) = \frac{x + (1 - \cos(2 \cdot \mathrm{frequency} \cdot x))}{2 \cdot \mathrm{frequency}}.
    $$

    See [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195).

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> tfa.activations.snake(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.29192656,  0.        ,  1.7080734 ], dtype=float32)>

    Args:
        x: A `Tensor`.
        frequency: A scalar, frequency of the periodic part.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)
    frequency = tf.cast(frequency, x.dtype)

    return x + (1 - tf.cos(2 * frequency * x)) / (2 * frequency)
