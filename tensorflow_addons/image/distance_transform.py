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
"""Distance transform ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils.resource_loader import get_path_to_datafile

_image_ops_so = tf.load_op_library(
    get_path_to_datafile("custom_ops/image/_image_ops.so"))

tf.no_gradient("EuclideanDistanceTransform")


@tf.function
def euclidean_dist_transform(images, dtype=tf.float32, name=None):
    """Applies euclidean distance transform(s) to the image(s).

    Args:
      images: A tensor of shape (num_images, num_rows, num_columns, 1) (NHWC),
        or (num_rows, num_columns, 1) (HWC). The rank must be statically known
        (the shape is not `TensorShape(None)`.
      dtype: DType of the output tensor.
      name: The name of the op.

    Returns:
      Image(s) with the type `dtype` and same shape as `images`, with the
      transform applied. If a tensor of all ones is given as input, the
      output tensor will be filled with the max value of the `dtype`.

    Raises:
      TypeError: If `image` is not tf.uint8, or `dtype` is not floating point.
      ValueError: If `image` more than one channel, or `image` is not of
        rank 3 or 4.
    """

    with tf.name_scope(name or "euclidean_distance_transform"):
        image_or_images = tf.convert_to_tensor(images, name="images")

        if image_or_images.dtype.base_dtype != tf.uint8:
            raise TypeError(
                "Invalid dtype %s. Expected uint8." % image_or_images.dtype)
        if image_or_images.get_shape().ndims is None:
            raise ValueError("`images` rank must be statically known")
        elif len(image_or_images.get_shape()) == 3:
            images = image_or_images[None, :, :, :]
        elif len(image_or_images.get_shape()) == 4:
            images = image_or_images
        else:
            raise ValueError("`images` should have rank between 3 and 4")

        if images.get_shape()[3] != 1 and images.get_shape()[3] is not None:
            raise ValueError("`images` must have only one channel")

        if dtype not in [tf.float16, tf.float32, tf.float64]:
            raise TypeError("`dtype` must be float16, float32 or float64")

        images = tf.cast(images, dtype)
        output = _image_ops_so.euclidean_distance_transform(images)

        if len(image_or_images.get_shape()) == 3:
            return output[0, :, :, :]
        return output
