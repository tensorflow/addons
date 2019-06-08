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


@tf.function
def _normalize(li, ma):
    one = tf.convert_to_tensor(1.0)
    two = tf.convert_to_tensor(255.0)

    def func1():
        return li

    def func2():
        return tf.math.truediv(li, two)

    return tf.cond(tf.math.greater(ma, one), func2, func1)


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
def median_filter2d(image, filter_shape=(3, 3), name=None):
    """This method performs Median Filtering on image. Filter shape can be user
    given.

    This method takes both kind of images where pixel values lie between 0 to
    255 and where it lies between 0.0 and 1.0
    Args:
        image: A 3D `Tensor` of type `float32` or 'int32' or 'float64' or
               'int64 and of shape`[rows, columns, channels]`

        filter_shape: Optional Argument. A tuple of 2 integers (R,C).
               R is the first value is the number of rows in the filter and
               C is the second value in the filter is the number of columns
               in the filter. This creates a filter of shape (R,C) or RxC
               filter. Default value = (3,3)
        name: The name of the op.

     Returns:
         A 3D median filtered image tensor of shape [rows,columns,channels] and
         type 'int32'. Pixel value of returned tensor ranges between 0 to 255
    """

    with tf.name_scope(name or "median_filter2d"):
        if not isinstance(filter_shape, tuple):
            raise TypeError('Filter shape must be a tuple')
        if len(filter_shape) != 2:
            raise ValueError('Filter shape must be a tuple of 2 integers. '
                             'Got %s values in tuple' % len(filter_shape))
        filter_shapex = filter_shape[0]
        filter_shapey = filter_shape[1]
        if not isinstance(filter_shapex, int) or not isinstance(
                filter_shapey, int):
            raise TypeError('Size of the filter must be Integers')
        (row, col, ch) = (image.shape[0], image.shape[1], image.shape[2])
        if row != None and col != None and ch != None:
            (row, col, ch) = (int(row), int(col), int(ch))
        else:
            raise TypeError('All the Dimensions of the input image '
                            'tensor must be Integers.')
        if row < filter_shapex or col < filter_shapey:
            raise ValueError(
                'Number of Pixels in each dimension of the image should be \
                more than the filter size. Got filter_shape (%sx' %
                filter_shape[0] + '%s).' % filter_shape[1] +
                ' Image Shape (%s)' % image.shape)
        if filter_shapex % 2 == 0 or filter_shapey % 2 == 0:
            raise ValueError('Filter size should be odd. Got filter_shape '
                             '(%sx%s)' % (filter_shape[0], filter_shape[1]))
        image = tf.cast(image, tf.float32)
        tf_i = tf.reshape(image, [row * col * ch])
        ma = tf.math.reduce_max(tf_i)
        image = _normalize(image, ma)

        # k and l is the Zero-padding size

        listi = []
        for a in range(ch):
            img = image[:, :, a:a + 1]
            img = tf.reshape(img, [1, row, col, 1])
            slic = tf.image.extract_patches(
                img, [1, filter_shapex, filter_shapey, 1], [1, 1, 1, 1],
                [1, 1, 1, 1],
                padding='SAME')
            mid = int(filter_shapex * filter_shapey / 2 + 1)
            top = tf.nn.top_k(slic, mid, sorted=True)
            li = tf.slice(top[0], [0, 0, 0, mid - 1], [-1, -1, -1, 1])
            li = tf.reshape(li, [row, col, 1])
            listi.append(li)
        y = tf.concat(listi[0], 2)

        for i in range(len(listi) - 1):
            y = tf.concat([y, listi[i + 1]], 2)

        y *= 255
        y = tf.cast(y, tf.int32)

        return y
