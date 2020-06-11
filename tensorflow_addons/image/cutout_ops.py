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
"""Cutout op"""

import tensorflow as tf
from tensorflow_addons.utils import keras_utils
from tensorflow_addons.utils.types import TensorLike, Number


def _get_image_wh(images, data_format):
    if data_format == "channels_last":
        image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
    else:
        image_height, image_width = tf.shape(images)[2], tf.shape(images)[3]

    return image_height, image_width


def _norm_params(images, mask_size, data_format):
    mask_size = tf.convert_to_tensor(mask_size)
    if tf.executing_eagerly():
        tf.assert_equal(
            tf.reduce_any(mask_size % 2 != 0),
            False,
            "mask_size should be divisible by 2",
        )
    if tf.rank(mask_size) == 0:
        mask_size = tf.stack([mask_size, mask_size])
    data_format = keras_utils.normalize_data_format(data_format)
    image_height, image_width = _get_image_wh(images, data_format)
    return mask_size, data_format, image_height, image_width


def random_cutout(
    images: TensorLike,
    mask_size: TensorLike,
    constant_values: Number = 0,
    seed: Number = None,
    data_format: str = "channels_last",
) -> tf.Tensor:
    """Apply cutout (https://arxiv.org/abs/1708.04552) to images.

    This operation applies a (mask_height x mask_width) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole images.

    Args:
      images: A tensor of shape
        (batch_size, height, width, channels)
        (NHWC), (batch_size, channels, height, width)(NCHW).
      mask_size: Specifies how big the zero mask that will be generated is that
        is applied to the images. The mask will be of size
        (mask_height x mask_width). Note: mask_size should be divisible by 2.
      constant_values: What pixel value to fill in the images in the area that has
        the cutout mask applied to it.
      seed: A Python integer. Used in combination with `tf.random.set_seed` to
        create a reproducible sequence of tensors across multiple calls.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
    Returns:
      An image Tensor.
    Raises:
      InvalidArgumentError: if mask_size can't be divisible by 2.
    """
    batch_size = tf.shape(images)[0]
    mask_size, data_format, image_height, image_width = _norm_params(
        images, mask_size, data_format
    )

    cutout_center_height = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32, seed=seed
    )
    cutout_center_width = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32, seed=seed
    )

    offset = tf.transpose([cutout_center_height, cutout_center_width], [1, 0])
    return cutout(images, mask_size, offset, constant_values, data_format,)


def cutout(
    images: TensorLike,
    mask_size: TensorLike,
    offset: TensorLike = (0, 0),
    constant_values: Number = 0,
    data_format: str = "channels_last",
) -> tf.Tensor:
    """Apply cutout (https://arxiv.org/abs/1708.04552) to images.

    This operation applies a (mask_height x mask_width) mask of zeros to
    a location within `img` specified by the offset. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole images.

    Args:
      images: A tensor of shape (batch_size, height, width, channels)
        (NHWC), (batch_size, channels, height, width)(NCHW).
      mask_size: Specifies how big the zero mask that will be generated is that
        is applied to the images. The mask will be of size
        (mask_height x mask_width). Note: mask_size should be divisible by 2.
      offset: A tuple of (height, width) or (batch_size, 2)
      constant_values: What pixel value to fill in the images in the area that has
        the cutout mask applied to it.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
    Returns:
      An image Tensor.
    Raises:
      InvalidArgumentError: if mask_size can't be divisible by 2.
    """
    with tf.name_scope("cutout"):
        origin_shape = images.shape
        offset = tf.convert_to_tensor(offset)
        mask_size, data_format, image_height, image_width = _norm_params(
            images, mask_size, data_format
        )
        mask_size = mask_size // 2

        if tf.rank(offset) == 1:
            offset = tf.expand_dims(offset, 0)
        cutout_center_heights = offset[:, 0]
        cutout_center_widths = offset[:, 1]

        lower_pads = tf.maximum(0, cutout_center_heights - mask_size[0])
        upper_pads = tf.maximum(0, image_height - cutout_center_heights - mask_size[0])
        left_pads = tf.maximum(0, cutout_center_widths - mask_size[1])
        right_pads = tf.maximum(0, image_width - cutout_center_widths - mask_size[1])

        cutout_shape = tf.transpose(
            [
                image_height - (lower_pads + upper_pads),
                image_width - (left_pads + right_pads),
            ],
            [1, 0],
        )
        masks = tf.TensorArray(images.dtype, 0, dynamic_size=True)
        for i in tf.range(tf.shape(cutout_shape)[0]):
            padding_dims = [
                [lower_pads[i], upper_pads[i]],
                [left_pads[i], right_pads[i]],
            ]
            mask = tf.pad(
                tf.zeros(cutout_shape[i], dtype=images.dtype),
                padding_dims,
                constant_values=1,
            )
            masks = masks.write(i, mask)

        if data_format == "channels_last":
            mask_4d = tf.expand_dims(masks.stack(), -1)
            mask = tf.tile(mask_4d, [1, 1, 1, tf.shape(images)[-1]])
        else:
            mask_4d = tf.expand_dims(masks.stack(), 1)
            mask = tf.tile(mask_4d, [1, tf.shape(images)[1], 1, 1])
        images = tf.where(
            mask == 0,
            tf.ones_like(images, dtype=images.dtype) * constant_values,
            images,
        )
        images.set_shape(origin_shape)
        return images
