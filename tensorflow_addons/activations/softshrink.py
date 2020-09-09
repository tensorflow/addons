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

import warnings
import tensorflow as tf
from tensorflow_addons.utils.types import Number

from tensorflow_addons.utils import types
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow_addons import options

_activation_so = LazySO("custom_ops/activations/_activation_ops.so")


@tf.keras.utils.register_keras_serializable(package="Addons")
def softshrink(
    x: types.TensorLike, lower: Number = -0.5, upper: Number = 0.5
) -> tf.Tensor:
    r"""Soft shrink function.

    Computes soft shrink function:

    $$
    \mathrm{softshrink}(x) =
    \begin{cases}
        x - \mathrm{lower} & \text{if } x < \mathrm{lower} \\
        x - \mathrm{upper} & \text{if } x > \mathrm{upper} \\
        0                  & \text{otherwise}
    \end{cases}.
    $$

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> tfa.activations.softshrink(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.5,  0. ,  0.5], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `float16`, `float32`, `float64`.
        lower: `float`, lower bound for setting values to zeros.
        upper: `float`, upper bound for setting values to zeros.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)

    if not options.TF_ADDONS_PY_OPS:
        try:
            return _softshrink_custom_op(x, lower, upper)
        except tf.errors.NotFoundError:
            options.warn_fallback("softshrink")

    return _softshrink_py(x, lower, upper)


def _softshrink_custom_op(x, lower, upper):
    warnings.warn(
        "The activations custom ops are deprecated and will be "
        "removed in TensorFlow Addons "
        "v0.12.0. \nPlease use the pure python version of softshrink instead "
        "by using the "
        "`TF_ADDONS_PY_OPS` flag. \nFor more info about this flag, see "
        "https://github.com/tensorflow/addons#gpucpu-custom-ops ",
        DeprecationWarning,
    )
    return _activation_so.ops.addons_softshrink(x, lower, upper)


@tf.RegisterGradient("Addons>Softshrink")
def _softshrink_grad(op, grad):
    return _activation_so.ops.addons_softshrink_grad(
        grad, op.inputs[0], op.get_attr("lower"), op.get_attr("upper")
    )


def _softshrink_py(x, lower, upper):
    if lower > upper:
        raise ValueError(
            "The value of lower is {} and should"
            " not be higher than the value "
            "variable upper, which is {} .".format(lower, upper)
        )
    values_below_lower = tf.where(x < lower, x - lower, 0)
    values_above_upper = tf.where(upper < x, x - upper, 0)
    return values_below_lower + values_above_upper
