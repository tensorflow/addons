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

    def _normalize(li):
        one = tf.convert_to_tensor(1.0)
        two = tf.convert_to_tensor(255.0)

        def func1():
            return li

        def func2():
            return tf.math.truediv(li, two)

        return tf.cond(tf.math.greater(ma, one), func2, func1)

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
                'No of Pixels in each dimension of the image should be more \
                than the filter size. Got filter_shape (%sx' % filter_shape[0]
                + '%s).' % filter_shape[1] + ' Image Shape (%s)' % image.shape)
        if filter_shapex % 2 == 0 or filter_shapey % 2 == 0:
            raise ValueError('Filter size should be odd. Got filter_shape '
                             '(%sx%s)' % (filter_shape[0], filter_shape[1]))
        image = tf.cast(image, tf.float32)
        tf_i = tf.reshape(image, [row * col * ch])
        ma = tf.math.reduce_max(tf_i)
        image = _normalize(image)

        # k and l is the Zero-padding size

        listi = []
        for a in range(ch):
            img = image[:, :, a:a + 1]
            img = tf.reshape(img, [1, row, col, 1])
            slic = tf.image.extract_image_patches(
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
