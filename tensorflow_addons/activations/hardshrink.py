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
def hardshrink(
    x: types.TensorLike, lower: Number = -0.5, upper: Number = 0.5
) -> tf.Tensor:
    r"""Hard shrink function.

    Computes hard shrink function:

    $$
    \mathrm{hardshrink}(x) =
    \begin{cases}
        x & \text{if } x < \text{lower} \\
        x & \text{if } x > \text{upper} \\
        0 & \text{otherwise}
    \end{cases}.
    $$

    Usage:

    >>> x = tf.constant([1.0, 0.0, 1.0])
    >>> tfa.activations.hardshrink(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1., 0., 1.], dtype=float32)>

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
            return _hardshrink_custom_op(x, lower, upper)
        except tf.errors.NotFoundError:
            options.warn_fallback("hardshrink")

    return _hardshrink_py(x, lower, upper)


def _hardshrink_custom_op(x, lower=-0.5, upper=0.5):
    """Alias with lazy loading of the .so file"""
    warnings.warn(
        "The activations custom ops are deprecated and will be removed in "
        "TensorFlow Addons v0.12.0. \nPlease use the pure python version of "
        "hardshrink instead by using the "
        "`TF_ADDONS_PY_OPS` flag. \nFor more info about this flag, see "
        "https://github.com/tensorflow/addons#gpucpu-custom-ops ",
        DeprecationWarning,
    )
    return _activation_so.ops.addons_hardshrink(x, lower, upper)


@tf.RegisterGradient("Addons>Hardshrink")
def _hardshrink_grad(op, grad):
    return _activation_so.ops.addons_hardshrink_grad(
        grad, op.inputs[0], op.get_attr("lower"), op.get_attr("upper")
    )


def _hardshrink_py(
    x: types.TensorLike, lower: Number = -0.5, upper: Number = 0.5
) -> tf.Tensor:
    if lower > upper:
        raise ValueError(
            "The value of lower is {} and should"
            " not be higher than the value "
            "variable upper, which is {} .".format(lower, upper)
        )
    mask_lower = x < lower
    mask_upper = upper < x
    mask = tf.logical_or(mask_lower, mask_upper)
    mask = tf.cast(mask, x.dtype)
    return x * mask
