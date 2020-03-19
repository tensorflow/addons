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
from tensorflow_addons.utils.types import TensorLike, Number
from tensorflow_addons.image.utils import from_4D_image, to_4D_image
from tensorflow.python.keras.utils import conv_utils


def _get_image_wh(images, data_format):
    if tf.equal(tf.rank(images), 4):
        if data_format == "channels_last":
            image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
        else:
            image_height, image_width = tf.shape(images)[2], tf.shape(images)[3]
    else:
        image_height, image_width = tf.shape(images)[0], tf.shape(images)[1]

    return image_height, image_width


def random_cutout(
    images: TensorLike,
    mask_size: TensorLike,
    constant_values: Number = 0,
    seed: Number = None,
    data_format: str = "channels_last",
) -> tf.Tensor:
    """Apply cutout (https://arxiv.org/abs/1708.04552) to images.

    This operation applies a (2*pad_size[0] x 2*pad_size[1]) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole images.

    Args:
      images: A tensor of shape
        (num_images, num_rows, num_columns, num_channels)
        (NHWC), (num_images, num_channels, num_rows, num_columns)(NCHW), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW).
      mask_size: Specifies how big the zero mask that will be generated is that
        is applied to the images. The mask will be of size
        (2*pad_size[0] x 2*pad_size[1]).
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
    """
    if tf.equal(tf.rank(mask_size), 0):
        mask_size = [mask_size, mask_size]
    data_format = conv_utils.normalize_data_format(data_format)
    # Sample the center location in the images where the zero mask will be applied.
    image_height, image_width = _get_image_wh(images, data_format)
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=image_height, dtype=tf.int32, seed=seed
    )
    cutout_center_width = tf.random.uniform(
        shape=[], minval=0, maxval=image_width, dtype=tf.int32, seed=seed
    )
    return cutout(
        images,
        mask_size,
        [cutout_center_height, cutout_center_width],
        constant_values,
        data_format,
    )


def cutout(
    images: TensorLike,
    mask_size: TensorLike,
    offset: TensorLike,
    constant_values: Number = 0,
    data_format: str = "channels_last",
) -> tf.Tensor:
    """Apply cutout (https://arxiv.org/abs/1708.04552) to images.

    This operation applies a (2*pad_size[0] x 2*pad_size[1]) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole images.

    Args:
      images: A tensor of shape
        (num_images, num_rows, num_columns, num_channels)
        (NHWC), (num_images, num_channels, num_rows, num_columns)(NCHW), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW).
      mask_size: Specifies how big the zero mask that will be generated is that
        is applied to the images. The mask will be of size
        (2*pad_size[0] x 2*pad_size[1]).
      offset: A tuple of (height, width)
      constant_values: What pixel value to fill in the images in the area that has
        the cutout mask applied to it.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
    Returns:
      An image Tensor.
    """
    with tf.name_scope("cutout"):
        if tf.equal(tf.rank(mask_size), 0):
            mask_size = [mask_size, mask_size]
        data_format = conv_utils.normalize_data_format(data_format)
        image_height, image_width = _get_image_wh(images, data_format)
        cutout_center_height = offset[0]
        cutout_center_width = offset[1]

        lower_pad = tf.maximum(0, cutout_center_height - mask_size[0])
        upper_pad = tf.maximum(0, image_height - cutout_center_height - mask_size[0])
        left_pad = tf.maximum(0, cutout_center_width - mask_size[1])
        right_pad = tf.maximum(0, image_width - cutout_center_width - mask_size[1])

        cutout_shape = [
            image_height - (lower_pad + upper_pad),
            image_width - (left_pad + right_pad),
        ]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = tf.pad(
            tf.zeros(cutout_shape, dtype=images.dtype), padding_dims, constant_values=1
        )
        mask_4d = to_4D_image(mask)
        if tf.equal(tf.rank(images), 3):
            mask = tf.tile(from_4D_image(mask_4d, 3), [1, 1, tf.shape(images)[-1]])
        elif tf.equal(tf.rank(images), 4):
            if data_format == "channels_last":
                mask = tf.tile(
                    mask_4d, [tf.shape(images)[0], 1, 1, tf.shape(images)[-1]]
                )
            else:
                mask = tf.tile(
                    mask_4d, [tf.shape(images)[0], tf.shape(images)[1], 1, 1]
                )
        images = tf.where(
            tf.equal(mask, 0),
            tf.ones_like(images, dtype=images.dtype) * constant_values,
            images,
        )
        return images
