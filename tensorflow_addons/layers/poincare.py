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
"""Implementing PoincareNormalize layer."""

import tensorflow as tf
from typeguard import typechecked
from typing import Union, List


@tf.keras.utils.register_keras_serializable(package="Addons")
class PoincareNormalize(tf.keras.layers.Layer):
    """Project into the Poincare ball with `norm <= 1.0 - epsilon`.

    See [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/pdf/1705.08039.pdf),
    and [wiki](https://en.wikipedia.org/wiki/Poincare_ball_model).

    For a 1-D tensor with `axis = 0`, computes

                  (x * (1 - epsilon)) / ||x||     if ||x|| > 1 - epsilon
        output =
                   x                              otherwise

    For `x` with more dimensions, independently normalizes each 1-D slice along
    dimension `axis`.

    Arguments:
      axis: Axis along which to normalize.  A scalar or a vector of integers.
      epsilon: A small deviation from the edge of the unit sphere for
        numerical stability.
    """

    @typechecked
    def __init__(
        self, axis: Union[None, int, List[int]] = 1, epsilon: float = 1e-5, **kwargs
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        square_sum = tf.math.reduce_sum(tf.math.square(x), self.axis, keepdims=True)
        x_inv_norm = tf.math.rsqrt(square_sum)
        x_inv_norm = tf.math.minimum((1.0 - self.epsilon) * x_inv_norm, 1.0)
        outputs = tf.math.multiply(x, x_inv_norm)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"axis": self.axis, "epsilon": self.epsilon}
        base_config = super().get_config()
        return {**base_config, **config}
