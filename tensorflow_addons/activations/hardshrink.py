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


def _hardshrink(x, lower, upper):
    mask_lower = x < lower
    mask_upper = upper < x
    mask = tf.logical_or(mask_lower, mask_upper)
    mask = tf.cast(mask, x.dtype)
    return x * mask


def compile_with_xla(func, dtype):
    compiled = tf.function(
        func,
        input_signature=(tf.TensorSpec(shape=None, dtype=dtype),
                         tf.TensorSpec(shape=tuple(), dtype=dtype),
                         tf.TensorSpec(shape=tuple(), dtype=dtype)),
        autograph=False,
        experimental_compile=True
    )
    return compiled


supported_dtypes = [tf.float16, tf.float32, tf.float64]

function_dispatch = {}
for dtype in supported_dtypes:
    function_dispatch[dtype] = compile_with_xla(_hardshrink, dtype)


@tf.keras.utils.register_keras_serializable(package='Addons')
def hardshrink(x, lower=-0.5, upper=0.5):
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
    if lower > upper:
        raise ValueError("The value of lower is {} and should"
                         " not be higher than the value "
                         "variable upper, which is {} .".format(lower, upper))
    x = tf.convert_to_tensor(x)
    return function_dispatch[x.dtype](x, lower, upper)
