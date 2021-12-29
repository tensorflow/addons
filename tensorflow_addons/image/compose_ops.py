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
"""Compose Ops"""

import tensorflow as tf

from tensorflow_addons.utils.types import TensorLike, Number


def blend(image1: TensorLike, image2: TensorLike, factor: Number) -> tf.Tensor:
    """Blend `image1` and `image2` using `factor`.

    Factor can be above 0.0.  A value of 0.0 means only `image1` is used.
    A value of 1.0 means only `image2` is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
      image1: An image Tensor of shape
          `(num_rows, num_columns, num_channels)` (HWC), or
          `(num_rows, num_columns)` (HW), or
          `(num_channels, num_rows, num_columns)` (CHW).
      image2: An image Tensor of shape
          `(num_rows, num_columns, num_channels)` (HWC), or
          `(num_rows, num_columns)` (HW), or
          `(num_channels, num_rows, num_columns)`.
      factor: A floating point value or Tensor of type `tf.float32` above 0.0.

    Returns:
      A blended image Tensor of `tf.float32`.

    """
    with tf.name_scope("blend"):

        if factor == 0.0:
            return tf.convert_to_tensor(image1)
        if factor == 1.0:
            return tf.convert_to_tensor(image2)

        image1 = tf.cast(image1, dtype=tf.dtypes.float32)
        image2 = tf.cast(image2, dtype=tf.dtypes.float32)

        difference = image2 - image1
        scaled = factor * difference

        # Do addition in float.
        temp = image1 + scaled

        # Interpolate
        if factor > 0.0 and factor < 1.0:
            # Interpolation means we always stay within 0 and 255.
            temp = tf.round(temp)
            return temp

        # Extrapolate:
        #
        # We need to clip and then cast.
        temp = tf.round(tf.clip_by_value(temp, 0.0, 255.0))
        return temp
