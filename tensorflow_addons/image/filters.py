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
from tensorflow_addons.image import utils as img_utils
from tensorflow_addons.utils import keras_utils
from tensorflow_addons.utils.types import TensorLike, FloatTensorLike

from typing import Optional, Union, List, Tuple


def _pad(
    image: TensorLike,
    filter_shape: Union[List[int], Tuple[int]],
    mode: str = "CONSTANT",
    constant_values: TensorLike = 0,
) -> tf.Tensor:
    """Explicitly pad a 4-D image.

    Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
    `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
    padding mode. For the even-sized filter, it pads one more value to the
    right or the bottom side.

    Args:
      image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: A `tuple`/`list` of 2 integers, specifying the height
        and width of the 2-D filter.
      mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
    """
    assert mode in ["CONSTANT", "REFLECT", "SYMMETRIC"]
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)


@tf.function
def mean_filter2d(
    image: TensorLike,
    filter_shape: Union[List[int], Tuple[int]] = [3, 3],
    padding: str = "REFLECT",
    constant_values: TensorLike = 0,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Perform mean filtering on image(s).

    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D mean filter. Can be a single integer
        to specify the same value for all spatial dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      2-D, 3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        or if `filter_shape` is invalid.
    """
    with tf.name_scope(name or "mean_filter2d"):
        image = tf.convert_to_tensor(image, name="image")
        original_ndims = img_utils.get_ndims(image)
        image = img_utils.to_4D_image(image)

        if padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError(
                'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
            )

        filter_shape = keras_utils.normalize_tuple(filter_shape, 2, "filter_shape")

        # Keep the precision if it's float;
        # otherwise, convert to float32 for computing.
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.dtypes.cast(image, tf.dtypes.float32)

        # Explicitly pad the image
        image = _pad(image, filter_shape, mode=padding, constant_values=constant_values)

        # Filter of shape (filter_width, filter_height, in_channels, 1)
        # has the value of 1 for each element.
        area = tf.constant(filter_shape[0] * filter_shape[1], dtype=image.dtype)
        filter_shape += (tf.shape(image)[-1], 1)
        kernel = tf.ones(shape=filter_shape, dtype=image.dtype)

        output = tf.nn.depthwise_conv2d(
            image, kernel, strides=(1, 1, 1, 1), padding="VALID"
        )

        output /= area

        output = img_utils.from_4D_image(output, original_ndims)
        return tf.dtypes.cast(output, orig_dtype)


@tf.function
def median_filter2d(
    image: TensorLike,
    filter_shape: Union[List[int], Tuple[int]] = [3, 3],
    padding: str = "REFLECT",
    constant_values: FloatTensorLike = 0,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Perform median filtering on image(s).

    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D median filter. Can be a single integer
        to specify the same value for all spatial dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      2-D, 3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        or if `filter_shape` is invalid.
    """
    with tf.name_scope(name or "median_filter2d"):
        image = tf.convert_to_tensor(image, name="image")
        original_ndims = img_utils.get_ndims(image)
        image = img_utils.to_4D_image(image)

        if padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError(
                'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
            )

        filter_shape = keras_utils.normalize_tuple(filter_shape, 2, "filter_shape")

        image_shape = tf.shape(image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        channels = image_shape[3]

        # Explicitly pad the image
        image = _pad(image, filter_shape, mode=padding, constant_values=constant_values)

        area = filter_shape[0] * filter_shape[1]

        floor = (area + 1) // 2
        ceil = area // 2 + 1

        patches = tf.image.extract_patches(
            image,
            sizes=[1, filter_shape[0], filter_shape[1], 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patches = tf.reshape(patches, shape=[batch_size, height, width, area, channels])

        patches = tf.transpose(patches, [0, 1, 2, 4, 3])

        # Note the returned median is casted back to the original type
        # Take [5, 6, 7, 8] for example, the median is (6 + 7) / 2 = 3.5
        # It turns out to be int(6.5) = 6 if the original type is int
        top = tf.nn.top_k(patches, k=ceil).values
        if area % 2 == 1:
            median = top[:, :, :, :, floor - 1]
        else:
            median = (top[:, :, :, :, floor - 1] + top[:, :, :, :, ceil - 1]) / 2

        output = tf.cast(median, image.dtype)
        output = img_utils.from_4D_image(output, original_ndims)
        return output


def _get_gaussian_kernel(sigma, filter_shape):
    """Compute 1D Gaussian kernel."""
    sigma = tf.convert_to_tensor(sigma)
    x = tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1)
    x = tf.cast(x ** 2, sigma.dtype)
    x = tf.exp(-x / (2.0 * (sigma ** 2)))
    x = x / tf.math.reduce_sum(x)
    return x


def _get_gaussian_kernel_2d(gaussian_filter_x, gaussian_filter_y):
    """Compute 2D Gaussian kernel given 1D kernels."""
    gaussian_kernel = tf.matmul(gaussian_filter_x, gaussian_filter_y)
    return gaussian_kernel


@tf.function
def gaussian_filter2d(
    image: FloatTensorLike,
    filter_shape: Union[List[int], Tuple[int]] = [3, 3],
    sigma: Union[List[float], Tuple[float]] = 1.0,
    padding: str = "REFLECT",
    constant_values: TensorLike = 0,
    name: Optional[str] = None,
) -> FloatTensorLike:
    """Perform Gaussian blur on image(s).

    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D gaussian filter. Can be a single
        integer to specify the same value for all spatial dimensions.
      sigma: A `float` or `tuple`/`list` of 2 floats, specifying
        the standard deviation in x and y direction the 2-D gaussian filter.
        Can be a single float to specify the same value for all spatial
        dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      2-D, 3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        if `filter_shape` is invalid,
        or if `sigma` is invalid.
    """
    with tf.name_scope(name or "gaussian_filter2d"):
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != 2:
                raise ValueError("sigma should be a float or a tuple/list of 2 floats")
        else:
            sigma = (sigma,) * 2

        if any(s < 0 for s in sigma):
            raise ValueError("sigma should be greater than or equal to 0.")

        if padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError(
                'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
            )

        image = tf.convert_to_tensor(image, name="image")
        sigma = tf.convert_to_tensor(sigma, name="sigma")

        original_ndims = img_utils.get_ndims(image)
        image = img_utils.to_4D_image(image)

        # Keep the precision if it's float;
        # otherwise, convert to float32 for computing.
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.cast(image, tf.float32)

        channels = tf.shape(image)[3]
        filter_shape = keras_utils.normalize_tuple(filter_shape, 2, "filter_shape")

        sigma = tf.cast(sigma, image.dtype)
        gaussian_kernel_x = _get_gaussian_kernel(sigma[1], filter_shape[1])
        gaussian_kernel_x = tf.reshape(gaussian_kernel_x, [1, filter_shape[1]])

        gaussian_kernel_y = _get_gaussian_kernel(sigma[0], filter_shape[0])
        gaussian_kernel_y = tf.reshape(gaussian_kernel_y, [filter_shape[0], 1])

        gaussian_kernel_2d = _get_gaussian_kernel_2d(
            gaussian_kernel_y, gaussian_kernel_x
        )
        gaussian_kernel_2d = tf.repeat(gaussian_kernel_2d, channels)
        gaussian_kernel_2d = tf.reshape(
            gaussian_kernel_2d, [filter_shape[0], filter_shape[1], channels, 1]
        )

        image = _pad(
            image, filter_shape, mode=padding, constant_values=constant_values,
        )

        output = tf.nn.depthwise_conv2d(
            input=image,
            filter=gaussian_kernel_2d,
            strides=(1, 1, 1, 1),
            padding="VALID",
        )
        output = img_utils.from_4D_image(output, original_ndims)
        return tf.cast(output, orig_dtype)
