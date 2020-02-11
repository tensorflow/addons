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
from tensorflow_addons.utils.resource_loader import LazySO

_activation_so = LazySO("custom_ops/activations/_activation_ops.so")


@tf.keras.utils.register_keras_serializable(package="Addons")
def hardshrink(
    x: types.TensorLike, lower: Number = -0.5, upper: Number = 0.5
) -> tf.Tensor:
    """Hard shrink function.

    Computes hard shrink function:
    `x if x < lower or x > upper else 0`.

    Args:
        x: A `Tensor`. Must be one of the following types:
            `float16`, `float32`, `float64`.
        lower: `float`, lower bound for setting values to zeros.
        upper: `float`, upper bound for setting values to zeros.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)
    return _activation_so.ops.addons_hardshrink(x, lower, upper)


@tf.RegisterGradient("Addons>Hardshrink")
def _hardshrink_grad(op, grad):
    return _activation_so.ops.addons_hardshrink_grad(
        grad, op.inputs[0], op.get_attr("lower"), op.get_attr("upper")
    )
