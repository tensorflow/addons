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
"""Color operations.
    equalize: Equalizes image histogram
    sharpness: Sharpen image
"""

import tensorflow as tf

from tensorflow_addons.utils.types import TensorLike, Number
from tensorflow_addons.image.utils import to_4D_image, from_4D_image
from tensorflow_addons.image.compose_ops import blend

from typing import Optional
from functools import partial


def equalize_image(image: TensorLike, data_format: str = "channels_last") -> tf.Tensor:
    """Implements Equalize function from PIL using TF ops."""

    @tf.function
    def scale_channel(image, channel):
        """Scale the data in the channel to implement equalize."""
        image_dtype = image.dtype

        if data_format == "channels_last":
            image = tf.cast(image[:, :, channel], tf.int32)
        elif data_format == "channels_first":
            image = tf.cast(image[channel], tf.int32)
        else:
            raise ValueError(
                "data_format can either be channels_last or channels_first"
            )
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(image, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.

        if step == 0:
            result = image
        else:
            result = tf.gather(build_lut(histo, step), image)

        return tf.cast(result, image_dtype)

    idx = 2 if data_format == "channels_last" else 0
    image = tf.stack([scale_channel(image, c) for c in range(image.shape[idx])], idx)

    return image


def equalize(
    image: TensorLike, data_format: str = "channels_last", name: Optional[str] = None
) -> tf.Tensor:
    """Equalize image(s)

    Args:
      images: A tensor of shape
          (num_images, num_rows, num_columns, num_channels) (NHWC), or
          (num_images, num_channels, num_rows, num_columns) (NCHW), or
          (num_rows, num_columns, num_channels) (HWC), or
          (num_channels, num_rows, num_columns) (CHW), or
          (num_rows, num_columns) (HW). The rank must be statically known (the
          shape is not `TensorShape(None)`).
      data_format: Either 'channels_first' or 'channels_last'
      name: The name of the op.
    Returns:
      Image(s) with the same type and shape as `images`, equalized.
    """
    with tf.name_scope(name or "equalize"):
        image_dims = tf.rank(image)
        image = to_4D_image(image)
        fn = partial(equalize_image, data_format=data_format)
        image = tf.map_fn(fn, image)
        return from_4D_image(image, image_dims)


def sharpness_image(image: TensorLike, factor: Number) -> tf.Tensor:
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image_dtype = image.dtype
    # SMOOTH PIL Kernel.
    image = tf.cast(image, tf.float32)
    kernel = (
        tf.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]
        )
        / 13.0
    )
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(
        image, kernel, strides, padding="VALID", dilations=[1, 1]
    )
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.cast(degenerate, image_dtype)

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[0, 0], [1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)
    # Blend the final result.
    blended = blend(result, orig_image, factor)
    return tf.cast(blended, image_dtype)


def sharpness(image: TensorLike, factor: Number) -> tf.Tensor:
    """Change sharpness of image(s).

    Args:
      images: A tensor of shape
          (num_images, num_rows, num_columns, num_channels) (NHWC), or
          (num_rows, num_columns, num_channels) (HWC)
      factor: A floating point value or Tensor above 0.0.
    Returns:
      Image(s) with the same type and shape as `images`, sharper.
    """
    image = tf.convert_to_tensor(image)
    image_dims = tf.rank(image)
    image = to_4D_image(image)
    image = sharpness_image(image, factor=factor)
    return from_4D_image(image, image_dims)
