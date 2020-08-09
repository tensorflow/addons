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

from tensorflow_addons.utils import types
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow_addons import options

_activation_so = LazySO("custom_ops/activations/_activation_ops.so")


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

    if not options.TF_ADDONS_PY_OPS:
        try:
            return _tanhshrink_custom_op(x)
        except tf.errors.NotFoundError:
            options.warn_fallback("tanhshrink")

    return _tanhshrink_py(x)


def _tanhshrink_custom_op(x):
    warnings.warn(
        "The activations custom ops are deprecated and will be removed in "
        "TensorFlow Addons v0.12.0. \nPlease use the pure python version of "
        "tanhshrink instead by using the "
        "`TF_ADDONS_PY_OPS` flag. \nFor more info about this flag, see "
        "https://github.com/tensorflow/addons#gpucpu-custom-ops ",
        DeprecationWarning,
    )
    return _activation_so.ops.addons_tanhshrink(x)


@tf.RegisterGradient("Addons>Tanhshrink")
def _tanhshrink_grad(op, grad):
    return _activation_so.ops.addons_tanhshrink_grad(grad, op.inputs[0])


def _tanhshrink_py(x):
    return x - tf.math.tanh(x)
