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
from tensorflow_addons.utils.resource_loader import LazySO

_activation_so = LazySO("custom_ops/activations/_activation_ops.so")


@tf.keras.utils.register_keras_serializable(package="Addons")
def gelu(x: types.TensorLike, approximate: bool = True) -> tf.Tensor:
    """Gaussian Error Linear Unit.

    Computes gaussian error linear:
    `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))` or
    `x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, where P(X) ~ N(0, 1),
    depending on whether approximation is enabled.

    See [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
    and [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

    Args:
        x: A `Tensor`. Must be one of the following types:
            `float16`, `float32`, `float64`.
        approximate: bool, whether to enable approximation.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)
    return _activation_so.ops.addons_gelu(x, approximate)


@tf.RegisterGradient("Addons>Gelu")
def _gelu_grad(op, grad):
    return _activation_so.ops.addons_gelu_grad(
        grad, op.inputs[0], op.get_attr("approximate")
    )
