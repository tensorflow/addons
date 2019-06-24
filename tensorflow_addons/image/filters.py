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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


def _pad(image, filter_shape, mode="CONSTANT", constant_values=0):
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
def mean_filter2d(image,
                  filter_shape=(3, 3),
                  padding="REFLECT",
                  constant_values=0,
                  name=None):
    """Perform mean filtering on image(s).

    Args:
      image: Either a 3-D `Tensor` of shape `[height, width, channels]`,
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
      3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        or if `filter_shape` is invalid.
    """
    with tf.name_scope(name or "mean_filter2d"):
        image = tf.convert_to_tensor(image, name="image")

        rank = image.shape.rank
        if rank != 3 and rank != 4:
            raise ValueError("image should be either 3 or 4-dimensional.")

        if padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError(
                "padding should be one of \"REFLECT\", \"CONSTANT\", or "
                "\"SYMMETRIC\".")

        filter_shape = keras_utils.conv_utils.normalize_tuple(
            filter_shape, 2, "filter_shape")

        # Expand to a 4-D tensor
        if rank == 3:
            image = tf.expand_dims(image, axis=0)

        # Keep the precision if it's float;
        # otherwise, convert to float32 for computing.
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.dtypes.cast(image, tf.dtypes.float32)

        # Explicitly pad the image
        image = _pad(
            image, filter_shape, mode=padding, constant_values=constant_values)

        # Filter of shape (filter_width, filter_height, in_channels, 1)
        # has the value of 1 for each element.
        area = tf.constant(
            filter_shape[0] * filter_shape[1], dtype=image.dtype)
        filter_shape = filter_shape + (tf.shape(image)[-1], 1)
        kernel = tf.ones(shape=filter_shape, dtype=image.dtype)

        output = tf.nn.depthwise_conv2d(
            image, kernel, strides=(1, 1, 1, 1), padding="VALID")

        output /= area

        # Squeeze out the first axis to make sure
        # output has the same dimension with image.
        if rank == 3:
            output = tf.squeeze(output, axis=0)

        return tf.dtypes.cast(output, orig_dtype)


@tf.function
def median_filter2d(image,
                    filter_shape=(3, 3),
                    padding="REFLECT",
                    constant_values=0,
                    name=None):
    """Perform median filtering on image(s).

    Args:
      image: Either a 3-D `Tensor` of shape `[height, width, channels]`,
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
      3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        or if `filter_shape` is invalid.
    """
    with tf.name_scope(name or "median_filter2d"):
        image = tf.convert_to_tensor(image, name="image")

        rank = image.shape.rank
        if rank != 3 and rank != 4:
            raise ValueError("image should be either 3 or 4-dimensional.")

        if padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError(
                "padding should be one of \"REFLECT\", \"CONSTANT\", or "
                "\"SYMMETRIC\".")

        filter_shape = keras_utils.conv_utils.normalize_tuple(
            filter_shape, 2, "filter_shape")

        # Expand to a 4-D tensor
        if rank == 3:
            image = tf.expand_dims(image, axis=0)

        image_shape = tf.shape(image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        channels = image_shape[3]

        # Explicitly pad the image
        image = _pad(
            image, filter_shape, mode=padding, constant_values=constant_values)

        area = filter_shape[0] * filter_shape[1]

        floor = (area + 1) // 2
        ceil = area // 2 + 1

        patches = tf.image.extract_patches(
            image,
            sizes=[1, filter_shape[0], filter_shape[1], 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID")

        patches = tf.reshape(
            patches, shape=[batch_size, height, width, area, channels])

        patches = tf.transpose(patches, [0, 1, 2, 4, 3])

        # Note the returned median is casted back to the original type
        # Take [5, 6, 7, 8] for example, the median is (6 + 7) / 2 = 3.5
        # It turns out to be int(6.5) = 6 if the original type is int
        top = tf.nn.top_k(patches, k=ceil).values
        if area % 2 == 1:
            median = top[:, :, :, :, floor - 1]
        else:
            median = (
                top[:, :, :, :, floor - 1] + top[:, :, :, :, ceil - 1]) / 2

        output = tf.cast(median, image.dtype)

        # Squeeze out the first axis to make sure
        # output has the same dimension with image.
        if rank == 3:
            output = tf.squeeze(output, axis=0)

        return output
