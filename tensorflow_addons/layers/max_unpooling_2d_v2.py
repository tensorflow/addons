# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""MaxUnpooling2DV2 operation."""

import tensorflow as tf

from typeguard import typechecked
from typing import Iterable

from tensorflow_addons.utils.keras_utils import normalize_tuple


def _max_unpooling_2d_v2(updates, mask, output_size):
    """Unpool the outputs of a maximum pooling operation."""
    mask = tf.cast(mask, "int32")
    input_shape = tf.shape(updates, out_type="int32")
    input_shape = [updates.shape[i] or input_shape[i] for i in range(4)]
    output_shape = output_size

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


@tf.keras.utils.register_keras_serializable(package="Addons")
class MaxUnpooling2DV2(tf.keras.layers.Layer):
    """Unpool the outputs of a maximum pooling operation.

    This differs from MaxUnpooling2D in that it uses output_size rather than strides and padding
    to calculate the unpooled tensor. This is because MaxPoolingWithArgMax can map several input
    sizes to the same output size, and specifying the output size avoids ambiguity in the
    inversion process.

    This function currently does not support outputs of MaxPoolingWithArgMax in following cases:
    - include_batch_in_index equals true.
    - The max pooling operation results in duplicate values in updates and mask.

    Args:
      output_size: A tuple/list of 4 integers specifying (batch_size, height, width, channel).
        The targeted output size.
    Call Args:
      updates: A 4D tensor of shape `(batch_size, height, width, channel)`.
        The pooling result from max pooling.
      mask: A 4D tensor of shape `(batch_size, height, width, channel)`.
        The indices of the maximal values.
    Output shape:
      4D tensor with the same shape as output_size.
    """

    @typechecked
    def __init__(
        self,
        output_size: Iterable[int],
        **kwargs,
    ):
        super(MaxUnpooling2DV2, self).__init__(**kwargs)

        self.output_size = normalize_tuple(output_size, 4, "output_size")

    def call(self, updates, mask):
        return _max_unpooling_2d_v2(updates, mask, output_size=self.output_size)

    def get_config(self):
        config = super(MaxUnpooling2DV2, self).get_config()
        config["output_size"] = self.output_size
        return config
