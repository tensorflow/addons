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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def poincare_normalize(x, axis=1, epsilon=1e-5, name=None):
    """Project into the Poincare ball with norm <= 1.0 - epsilon.

    https://en.wikipedia.org/wiki/Poincare_ball_model

    Used in
    Poincare Embeddings for Learning Hierarchical Representations
    Maximilian Nickel, Douwe Kiela
    https://arxiv.org/pdf/1705.08039.pdf

    For a 1-D tensor with `axis = 0`, computes

                  (x * (1 - epsilon)) / ||x||     if ||x|| > 1 - epsilon
        output =
                   x                              otherwise

    For `x` with more dimensions, independently normalizes each 1-D slice along
    dimension `axis`.

    Args:
      x: A `Tensor`.
      axis: Axis along which to normalize.  A scalar or a vector of
        integers.
      epsilon: A small deviation from the edge of the unit sphere for numerical
        stability.
      name: A name for this operation (optional).

    Returns:
      A `Tensor` with the same shape as `x`.
    """
    with ops.name_scope(name, 'poincare_normalize', [x]) as name:
        x = ops.convert_to_tensor(x, name='x')
        square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keepdims=True)
        x_inv_norm = math_ops.rsqrt(square_sum)
        x_inv_norm = math_ops.minimum((1. - epsilon) * x_inv_norm, 1.)
        return math_ops.multiply(x, x_inv_norm, name=name)
