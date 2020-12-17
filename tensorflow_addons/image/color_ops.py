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


def _scale_channel(image: TensorLike, channel: int) -> tf.Tensor:
    """Scale the data in the channel to implement equalize."""
    image_dtype = image.dtype
    image = tf.cast(image[:, :, channel], tf.int32)

    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(image, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero_histo = tf.boolean_mask(histo, histo != 0)
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    if step == 0:
        result = image
    else:
        lut_values = (tf.cumsum(histo, exclusive=True) + (step // 2)) // step
        lut_values = tf.clip_by_value(lut_values, 0, 255)
        result = tf.gather(lut_values, image)

    return tf.cast(result, image_dtype)


def _equalize_image(image: TensorLike) -> tf.Tensor:
    """Implements Equalize function from PIL using TF ops."""
    image = tf.stack([_scale_channel(image, c) for c in range(image.shape[-1])], -1)
    return image


@tf.function
def equalize(image: TensorLike, name: Optional[str] = None) -> tf.Tensor:
    """Equalize image(s)

    Args:
      images: A tensor of shape
          `(num_images, num_rows, num_columns, num_channels)` (NHWC), or
          `(num_rows, num_columns, num_channels)` (HWC), or
          `(num_rows, num_columns)` (HW). The rank must be statically known (the
          shape is not `TensorShape(None)`).
      name: The name of the op.
    Returns:
      Image(s) with the same type and shape as `images`, equalized.
    """
    with tf.name_scope(name or "equalize"):
        image_dims = tf.rank(image)
        image = to_4D_image(image)
        fn = partial(_equalize_image)
        image = tf.map_fn(fn, image)
        return from_4D_image(image, image_dims)


def _sharpness_image(image: TensorLike, factor: Number) -> tf.Tensor:
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image_dtype = image.dtype
    image_channels = image.shape[-1]
    image = tf.cast(image, tf.float32)

    # SMOOTH PIL Kernel.
    kernel = (
        tf.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]
        )
        / 13.0
    )
    kernel = tf.tile(kernel, [1, 1, image_channels, 1])

    # Apply kernel channel-wise.
    degenerate = tf.nn.depthwise_conv2d(
        image, kernel, strides=[1, 1, 1, 1], padding="VALID", dilations=[1, 1]
    )
    degenerate = tf.cast(degenerate, image_dtype)

    # For the borders of the resulting image, fill in the values of the original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[0, 0], [1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    blended = blend(result, orig_image, factor)
    return tf.cast(blended, image_dtype)


@tf.function
def sharpness(
    image: TensorLike, factor: Number, name: Optional[str] = None
) -> tf.Tensor:
    """Change sharpness of image(s).

    Args:
      image: A tensor of shape
          `(num_images, num_rows, num_columns, num_channels)` (NHWC), or
          `(num_rows, num_columns, num_channels)` (HWC)
      factor: A floating point value or Tensor above 0.0.
      name: The name of the op.
    Returns:
      Image(s) with the same type and shape as `images`, sharper.
    """
    with tf.name_scope(name or "sharpness"):
        image_dims = tf.rank(image)
        image = to_4D_image(image)
        image = _sharpness_image(image, factor=factor)
        return from_4D_image(image, image_dims)
