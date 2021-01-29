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
    clahe: Contrast-limited adaptive histogram equalization
"""

import tensorflow as tf

from tensorflow_addons.utils.types import TensorLike, Number
from tensorflow_addons.image.utils import to_4D_image, from_4D_image
from tensorflow_addons.image.compose_ops import blend

from typing import Optional, Tuple, Union, List
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


def _clahe(
    image: TensorLike, clip_limit: Number, tile_grid_size: Union[List[int], Tuple[int]]
) -> tf.Tensor:
    """Implements CLAHE as tf ops"""
    original_2d_shape = (tf.shape(image)[0], tf.shape(image)[1])
    original_dtype = image.dtype

    # Need image in int32 format for later gather_nd ops
    image = tf.cast(image, tf.int32)

    tile_shape = tf.truediv(original_2d_shape, tile_grid_size)
    tile_shape = tf.cast(tf.math.ceil(tile_shape), tf.int32)

    # Reflection-pad image
    pad_y = 0
    pad_x = 0

    if original_2d_shape[0] % tile_shape[0] != 0:
        pad_y = tile_shape[0] - (original_2d_shape[0] % tile_shape[0])

    if original_2d_shape[1] % tile_shape[1] != 0:
        pad_x = tile_shape[1] - (original_2d_shape[1] % tile_shape[1])

    image_padded = tf.pad(image, [[0, pad_y], [0, pad_x], [0, 0]], "REFLECT")

    all_tiles = tf.space_to_batch(
        input=tf.expand_dims(image_padded, axis=0),
        block_shape=tile_shape,
        paddings=[[0, 0], [0, 0]],
    )

    # Compute per-tile histogram
    single_dimension_tiles = tf.reshape(
        all_tiles,
        (
            tile_shape[0] * tile_shape[1],
            tile_grid_size[0] * tile_grid_size[1] * tf.shape(image)[-1],
        ),
    )

    single_dimension_tiles = tf.transpose(single_dimension_tiles)
    hists = tf.math.bincount(
        single_dimension_tiles, minlength=256, maxlength=256, axis=-1
    )

    hists = tf.transpose(hists)
    hists = tf.reshape(
        hists, (256, tile_grid_size[0], tile_grid_size[1], tf.shape(image)[-1])
    )

    # Clip histograms, if necessary
    if clip_limit > 0:
        clip_limit_actual = tf.cast(
            clip_limit * ((tile_shape[0] * tile_shape[1]) / 256), tf.int32
        )
        clipped_hists = tf.clip_by_value(
            hists, clip_value_min=0, clip_value_max=clip_limit_actual
        )
        clipped_px_count = tf.math.reduce_sum(hists - clipped_hists, axis=0)
        clipped_hists = tf.cast(clipped_hists, tf.float32)
        clipped_px_count = tf.cast(clipped_px_count, tf.float32)
        clipped_hists = clipped_hists + tf.math.truediv(clipped_px_count, 256)
    else:
        clipped_hists = tf.cast(hists, tf.float32)

    cdf = tf.math.cumsum(clipped_hists, axis=0)
    cdf_min = tf.math.reduce_min(cdf, axis=0)

    numerator = cdf - cdf_min
    denominator = tf.cast(tile_shape[0] * tile_shape[1], tf.float32) - cdf_min

    cdf_normalized = tf.round(tf.math.divide_no_nan(numerator, denominator) * (255))
    cdf_normalized = tf.cast(cdf_normalized, tf.int32)

    # Reflection-pad the cdf functions so that we don't have to explicitly deal with corners/edges
    cdf_padded = tf.pad(
        cdf_normalized, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC"
    )

    coords = tf.stack(
        tf.meshgrid(
            tf.range(tf.shape(image_padded)[0]),
            tf.range(tf.shape(image_padded)[1]),
            tf.range(tf.shape(image_padded)[2]),
            indexing="ij",
        )
    )

    y_coords = coords[0, :, :]
    x_coords = coords[1, :, :]
    z_coords = coords[2, :, :]

    half_tile_shape = tf.math.floordiv(tile_shape, 2)

    nw_y_component = tf.math.floordiv(y_coords - half_tile_shape[0], tile_shape[0])
    nw_x_component = tf.math.floordiv(x_coords - half_tile_shape[1], tile_shape[1])

    # Need to correct negative values because negative-indexing for gather_nd ops
    # not supported on all processors (cdf is padded to account for this)
    nw_y_component = nw_y_component + 1
    nw_x_component = nw_x_component + 1

    ne_y_component = nw_y_component
    ne_x_component = nw_x_component + 1

    sw_y_component = nw_y_component + 1
    sw_x_component = nw_x_component

    se_y_component = sw_y_component
    se_x_component = sw_x_component + 1

    def cdf_transform(x_comp, y_comp):
        gatherable = tf.stack([image_padded, y_comp, x_comp, z_coords], axis=-1)
        return tf.cast(tf.gather_nd(cdf_padded, gatherable), tf.float32)

    nw_transformed = cdf_transform(nw_x_component, nw_y_component)
    ne_transformed = cdf_transform(ne_x_component, ne_y_component)
    sw_transformed = cdf_transform(sw_x_component, sw_y_component)
    se_transformed = cdf_transform(se_x_component, se_y_component)

    a = (y_coords - half_tile_shape[0]) % tile_shape[0]
    a = tf.cast(tf.math.truediv(a, tile_shape[0]), tf.float32)
    b = (x_coords - half_tile_shape[1]) % tile_shape[1]
    b = tf.cast(tf.math.truediv(b, tile_shape[1]), tf.float32)

    # Interpolate
    interpolated = (a * (b * se_transformed + (1 - b) * sw_transformed)) + (1 - a) * (
        b * ne_transformed + (1 - b) * nw_transformed
    )

    # Return image to original size and dtype
    interpolated = interpolated[0 : original_2d_shape[0], 0 : original_2d_shape[1], :]
    interpolated = tf.cast(tf.round(interpolated), original_dtype)

    return interpolated


@tf.function
def clahe(
    image: TensorLike,
    clip_limit: Number = 4.0,
    tile_grid_size: Union[List[int], Tuple[int]] = (8, 8),
    name: Optional[str] = None,
) -> tf.Tensor:
    """
    Args:
        image: A tensor of shape
            `(num_images, num_rows, num_columns, num_channels)` or
            `(num_rows, num_columns, num_channels)`
        clip_limit: A floating point value or Tensor.
            0 will result in no clipping (AHE only).
            Limits the noise amplification in near-constant regions.
            Default 4.0.
        tile_grid_size: A tensor of shape
            `(tiles_in_x_direction, tiles_in_y_direction)`
            Specifies how many tiles to break the image into.
            Default (8x8).
        name: (Optional) The name of the op. Default `None`.
    Returns:
        Contrast-limited, adaptive-histogram-equalized image
    """
    with tf.name_scope(name or "clahe"):
        image_dims = tf.rank(image)
        image = to_4D_image(image)
        fn = partial(lambda x: _clahe(x, clip_limit, tile_grid_size))
        image = tf.map_fn(fn, image)
        return from_4D_image(image, image_dims)
