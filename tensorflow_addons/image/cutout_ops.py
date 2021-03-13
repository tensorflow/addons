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


@tf.function
def _norm_params(mask_size, offset=None):
    tf.assert_equal(
        tf.reduce_any(mask_size % 2 != 0),
        False,
        "mask_size should be divisible by 2",
    )
    if tf.rank(mask_size) == 0:
        mask_size = tf.stack([mask_size, mask_size])
    if offset is not None and tf.rank(offset) == 1:
        offset = tf.expand_dims(offset, 0)
    return mask_size, offset


@tf.function
def _random_center(mask_dim_length, image_dim_length, batch_size, seed):
    if mask_dim_length >= image_dim_length:
        return tf.tile([image_dim_length // 2], [batch_size])
    half_mask_dim_length = mask_dim_length // 2
    return tf.random.uniform(
        shape=[batch_size],
        minval=half_mask_dim_length,
        maxval=image_dim_length - half_mask_dim_length,
        dtype=tf.int32,
        seed=seed,
    )


def random_cutout(
    images: TensorLike,
    mask_size: TensorLike,
    constant_values: Number = 0,
    seed: Number = None,
) -> tf.Tensor:
    """Apply [cutout](https://arxiv.org/abs/1708.04552) to images with random offset.

    This operation applies a `(mask_height x mask_width)` mask of zeros to
    a random location within `images`. The pixel values filled in will be of
    the value `constant_values`. The location where the mask will be applied is
    randomly chosen uniformly over the whole images.

    Args:
      images: A tensor of shape `(batch_size, height, width, channels)` (NHWC).
      mask_size: Specifies how big the zero mask that will be generated is that
        is applied to the images. The mask will be of size
        `(mask_height x mask_width)`. Note: mask_size should be divisible by 2.
      constant_values: What pixel value to fill in the images in the area that has
        the cutout mask applied to it.
      seed: A Python integer. Used in combination with `tf.random.set_seed` to
        create a reproducible sequence of tensors across multiple calls.
    Returns:
      A `Tensor` of the same shape and dtype as `images`.
    Raises:
      InvalidArgumentError: if `mask_size` can't be divisible by 2.
    """
    images = tf.convert_to_tensor(images)
    mask_size = tf.convert_to_tensor(mask_size)

    image_dynamic_shape = tf.shape(images)
    batch_size, image_height, image_width = (
        image_dynamic_shape[0],
        image_dynamic_shape[1],
        image_dynamic_shape[2],
    )

    mask_size, _ = _norm_params(mask_size, offset=None)

    cutout_center_height = _random_center(mask_size[0], image_height, batch_size, seed)
    cutout_center_width = _random_center(mask_size[1], image_width, batch_size, seed)

    offset = tf.transpose([cutout_center_height, cutout_center_width], [1, 0])
    return cutout(images, mask_size, offset, constant_values)


def cutout(
    images: TensorLike,
    mask_size: TensorLike,
    offset: TensorLike = (0, 0),
    constant_values: Number = 0,
) -> tf.Tensor:
    """Apply [cutout](https://arxiv.org/abs/1708.04552) to images.

    This operation applies a `(mask_height x mask_width)` mask of zeros to
    a location within `images` specified by the offset.
    The pixel values filled in will be of the value `constant_values`.
    The location where the mask will be applied is randomly
    chosen uniformly over the whole images.

    Args:
      images: A tensor of shape `(batch_size, height, width, channels)` (NHWC).
      mask_size: Specifies how big the zero mask that will be generated is that
        is applied to the images. The mask will be of size
        `(mask_height x mask_width)`. Note: mask_size should be divisible by 2.
      offset: A tuple of `(height, width)` or `(batch_size, 2)`
      constant_values: What pixel value to fill in the images in the area that has
        the cutout mask applied to it.
    Returns:
      A `Tensor` of the same shape and dtype as `images`.
    Raises:
      InvalidArgumentError: if `mask_size` can't be divisible by 2.
    """
    with tf.name_scope("cutout"):
        images = tf.convert_to_tensor(images)
        mask_size = tf.convert_to_tensor(mask_size)
        offset = tf.convert_to_tensor(offset)

        image_static_shape = images.shape
        image_dynamic_shape = tf.shape(images)
        image_height, image_width, channels = (
            image_dynamic_shape[1],
            image_dynamic_shape[2],
            image_dynamic_shape[3],
        )

        mask_size, offset = _norm_params(mask_size, offset)
        mask_size = mask_size // 2

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

        def fn(i):
            padding_dims = [
                [lower_pads[i], upper_pads[i]],
                [left_pads[i], right_pads[i]],
            ]
            mask = tf.pad(
                tf.zeros(cutout_shape[i], dtype=tf.bool),
                padding_dims,
                constant_values=True,
            )
            return mask

        mask = tf.map_fn(
            fn,
            tf.range(tf.shape(cutout_shape)[0]),
            fn_output_signature=tf.TensorSpec(
                shape=image_static_shape[1:-1], dtype=tf.bool
            ),
        )
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 1, channels])

        images = tf.where(
            mask,
            images,
            tf.cast(constant_values, dtype=images.dtype),
        )
        images.set_shape(image_static_shape)
        return images
