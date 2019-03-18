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
from tensorflow.python.eager import def_function
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from tensorflow_addons.utils.resource_loader import get_path_to_datafile

_image_ops_so = tf.load_op_library(
    get_path_to_datafile("custom_ops/image/_image_ops.so"))

ops.NotDifferentiable("EuclideanDistanceTransform")
ops.RegisterShape("EuclideanDistanceTransform")(
    common_shapes.call_cpp_shape_fn)

_OUTPUT_DTYPES = [dtypes.float16, dtypes.float32, dtypes.float64]


@def_function.function
def euclidean_dist_transform(images, dtype=dtypes.float32, name=None):
    """
    Applies euclidean distance transform to the images_t

    Args:
      images: Tensor of shape (num_images, num_rows, num_columns, num_channels)
        (NHWC) or (num_rows, num_columns, num_channels) (HWC). The rank must be
        statically knownself. The image must be a binary image of uint8 type.
      dtype: The dtype of the output, must be float16, float32 or float64
      name: The name of the op.


    Returns:
      Image(s) with the same type and shape as `images`, with euclidean
      distance transform applied.

    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If `image` is not a binary image

    """
    with ops.name_scope(name, "euclidean_distance_transform", [images]):
        image_or_images = ops.convert_to_tensor(images, name="images")

        if image_or_images.dtype.base_dtype != dtypes.uint8:
            raise TypeError(
                "Invalid dtype %s. Excepted uint8." % image_or_images.dtype)
        elif image_or_images.get_shape().ndims is None:
            raise TypeError("`images` rank must be statically known")
        elif len(image_or_images.get_shape()) == 3:
            images = image_or_images[None, :, :, :]
        elif len(image_or_images.get_shape()) == 4:
            images = image_or_images
        else:
            raise TypeError("`images` should have rank between 3 and 4")

        if images.get_shape()[3] != 1:
            raise ValueError("`images` must be a binary image with 1 channel")

        if dtype not in _OUTPUT_DTYPES:
            raise ValueError("`dtype` must be float16, float32 or float64")

        images = math_ops.cast(images, dtype)
        output = _image_ops_so.euclidean_distance_transform(images)

        if len(image_or_images.get_shape()) == 3:
            return output[0, :, :, :]
        return output
