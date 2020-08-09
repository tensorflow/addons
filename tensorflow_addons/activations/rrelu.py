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
from tensorflow_addons.utils.types import TensorLike, Number
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
def rrelu(
    x: TensorLike,
    lower: Number = 0.125,
    upper: Number = 0.3333333333333333,
    training: Optional[bool] = None,
    seed: Optional[int] = None,
    rng: Optional[tf.random.Generator] = None,
) -> tf.Tensor:
    r"""Randomized leaky rectified liner unit function.

    Computes rrelu function:

    $$
    \mathrm{rrelu}(x) =
    \begin{cases}
        x & \text{if } x > 0 \\
        a x
    \end{cases},
    $$

    where

    $$
    a \sim \mathcal{U}(\mathrm{lower}, \mathrm{upper})
    $$

    when `training` is `True`; or

    $$
    a = \frac{\mathrm{lower} + \mathrm{upper}}{2}
    $$

    when `training` is `False`.

    See [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853).

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> tfa.activations.rrelu(x, training=False)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.22916667,  0.        ,  1.        ], dtype=float32)>
    >>> tfa.activations.rrelu(x, training=True, seed=2020)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.22631127,  0.        ,  1.        ], dtype=float32)>
    >>> generator = tf.random.Generator.from_seed(2021)
    >>> tfa.activations.rrelu(x, training=True, rng=generator)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.16031083,  0.        ,  1.        ], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `float16`, `float32`, `float64`.
        lower: `float`, lower bound for random alpha.
        upper: `float`, upper bound for random alpha.
        training: `bool`, indicating whether the `call`
        is meant for training or inference.
        seed: `int`, this sets the operation-level seed.
        rng: A `tf.random.Generator`.
    Returns:
        result: A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)
    lower = tf.cast(lower, x.dtype)
    upper = tf.cast(upper, x.dtype)

    def random_a():
        if rng is not None and seed is not None:
            raise ValueError(
                "Either seed or rng should be specified. Not both at the same time."
            )

        if rng is not None:
            return rng.uniform(tf.shape(x), minval=lower, maxval=upper, dtype=x.dtype)

        return tf.random.uniform(
            tf.shape(x), minval=lower, maxval=upper, dtype=x.dtype, seed=seed
        )

    a = tf.keras.backend.in_train_phase(random_a, (lower + upper) / 2, training)

    return tf.where(x >= 0, x, a * x)
