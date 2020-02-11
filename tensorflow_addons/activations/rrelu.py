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
from tensorflow_addons.utils.types import Number
from tensorflow_addons.utils import types
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
def rrelu(
    x: types.TensorLike,
    lower: Number = 0.125,
    upper: Number = 0.3333333333333333,
    training: Optional[str] = None,
    seed: Optional[str] = None,
) -> tf.Tensor:
    """rrelu function.

    Computes rrelu function:
    `x if x > 0 else random(lower, upper) * x` or
    `x if x > 0 else x * (lower + upper) / 2`
    depending on whether training is enabled.

    See [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853).

    Args:
        x: A `Tensor`. Must be one of the following types:
            `float16`, `float32`, `float64`.
        lower: `float`, lower bound for random alpha.
        upper: `float`, upper bound for random alpha.
        training: `bool`, indicating whether the `call`
        is meant for training or inference.
        seed: `int`, this sets the operation-level seed.
    Returns:
        result: A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)
    if training is None:
        training = tf.keras.backend.learning_phase()
        training = bool(tf.keras.backend.get_value(training))

    if training:
        alpha = tf.random.uniform(
            tf.shape(x), minval=lower, maxval=upper, dtype=x.dtype, seed=seed
        )
    else:
        alpha = tf.cast((lower + upper) / 2, x.dtype)

    return tf.where(x >= 0, x, alpha * x)
