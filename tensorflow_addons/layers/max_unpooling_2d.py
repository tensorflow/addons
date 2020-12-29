# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""MaxUnpooling2D operation."""

import tensorflow as tf

from typeguard import typechecked
from typing import Union

from tensorflow_addons.utils.keras_utils import normalize_tuple


def _calculate_output_shape(input_shape, pool_size, strides, padding):
    """Calculates the shape of the unpooled output."""
    if padding == "VALID":
        output_shape = (
            input_shape[0],
            (input_shape[1] - 1) * strides[0] + pool_size[0],
            (input_shape[2] - 1) * strides[1] + pool_size[1],
            input_shape[3],
        )
    elif padding == "SAME":
        output_shape = (
            input_shape[0],
            input_shape[1] * strides[0],
            input_shape[2] * strides[1],
            input_shape[3],
        )
    else:
        raise ValueError('Padding must be a string from: "SAME", "VALID"')
    return output_shape


def _max_unpooling_2d(updates, mask, pool_size=(2, 2), strides=(2, 2), padding="SAME"):
    """Unpool the outputs of a maximum pooling operation."""
    pool_size_attr = " ".join(["i: %d" % v for v in pool_size])
    strides_attr = " ".join(["i: %d" % v for v in strides])
    experimental_implements = [
        'name: "addons:MaxUnpooling2D"',
        'attr { key: "pool_size" value { list {%s} } }' % pool_size_attr,
        'attr { key: "strides" value { list {%s} } }' % strides_attr,
        'attr { key: "padding" value { s: "%s" } }' % padding,
    ]
    experimental_implements = " ".join(experimental_implements)

    @tf.function(experimental_implements=experimental_implements)
    def func(updates, mask):
        mask = tf.cast(mask, "int32")
        input_shape = tf.shape(updates, out_type="int32")
        input_shape = [updates.shape[i] or input_shape[i] for i in range(4)]
        output_shape = _calculate_output_shape(input_shape, pool_size, strides, padding)

        # Calculates indices for batch, height, width and feature maps.
        one_like_mask = tf.ones_like(mask, dtype="int32")
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype="int32"), shape=batch_shape
        )
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype="int32")
        f = one_like_mask * feature_range

        # Transposes indices & reshape update values to one dimension.
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

    return func(updates, mask)


@tf.keras.utils.register_keras_serializable(package="Addons")
class MaxUnpooling2D(tf.keras.layers.Layer):
    """Unpool the outputs of a maximum pooling operation.

    This function currently does not support outputs of MaxPoolingWithArgMax in
    following cases:
    - include_batch_in_index equals true.
    - input_shape is not divisible by strides if padding is "SAME".
    - (input_shape - pool_size) is not divisible by strides if padding is "VALID".
    - The max pooling operation results in duplicate values in updates and mask.

    Args:
      updates: The pooling result from max pooling.
      mask: the argmax result corresponds to above max values.
      pool_size: The filter that max pooling was performed with. Default: (2, 2).
      strides: The strides that max pooling was performed with. Default: (2, 2).
      padding: The padding that max pooling was performed with. Default: "SAME".
    Input shape:
      4D tensor with shape: `(batch_size, height, width, channel)`.
    Output shape:
      4D tensor with the same shape as the input of max pooling operation.
    """

    @typechecked
    def __init__(
        self,
        pool_size: Union[list, tuple, int] = (2, 2),
        strides: Union[list, tuple, int] = (2, 2),
        padding: str = "SAME",
        **kwargs,
    ):
        super(MaxUnpooling2D, self).__init__(**kwargs)

        if padding != "SAME" and padding != "VALID":
            raise ValueError('Padding must be a string from: "SAME", "VALID"')

        self.pool_size = normalize_tuple(pool_size, 2, "pool_size")
        self.strides = normalize_tuple(strides, 2, "strides")
        self.padding = padding

    def call(self, updates, mask):
        return _max_unpooling_2d(
            updates,
            mask,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[1]
        return _calculate_output_shape(
            input_shape, self.pool_size, self.strides, self.padding
        )

    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config["pool_size"] = self.pool_size
        config["strides"] = self.strides
        config["padding"] = self.padding
        return config
