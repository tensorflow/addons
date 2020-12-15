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
import math
import warnings

from tensorflow_addons.utils import types
from distutils.version import LooseVersion


@tf.keras.utils.register_keras_serializable(package="Addons")
def gelu(x: types.TensorLike, approximate: bool = True) -> tf.Tensor:
    r"""Gaussian Error Linear Unit.

    Computes gaussian error linear:

    $$
    \mathrm{gelu}(x) = x \Phi(x),
    $$

    where

    $$
    \Phi(x) = \frac{1}{2} \left[ 1 + \mathrm{erf}(\frac{x}{\sqrt{2}}) \right]$
    $$

    when `approximate` is `False`; or

    $$
    \Phi(x) = \frac{x}{2} \left[ 1 + \tanh(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)) \right]
    $$

    when `approximate` is `True`.

    See [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
    and [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

    Note that `approximate` will default to `False` from TensorFlow version 2.4 onwards.
    Consider using `tf.nn.gelu` instead.

    Usage:

    >>> tfa.options.TF_ADDONS_PY_OPS = True
    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> tfa.activations.gelu(x, approximate=False)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.15865529,  0.        ,  0.8413447 ], dtype=float32)>
    >>> tfa.activations.gelu(x, approximate=True)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.15880796,  0.        ,  0.841192  ], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `float16`, `float32`, `float64`.
        approximate: bool, whether to enable approximation.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    warnings.warn(
        "gelu activation has been migrated to core TensorFlow, "
        "and will be deprecated in Addons 0.13.",
        DeprecationWarning,
    )

    x = tf.convert_to_tensor(x)

    if LooseVersion(tf.__version__) >= "2.4":
        gelu_op = tf.nn.gelu
        warnings.warn(
            "Default value of `approximate` is changed from `True` to `False`"
        )
    else:
        gelu_op = _gelu_py

    return gelu_op(x, approximate)


def _gelu_py(x: types.TensorLike, approximate: bool = True) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    if approximate:
        pi = tf.cast(math.pi, x.dtype)
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))
