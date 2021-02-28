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

import tensorflow as tf
from tensorflow_addons.image import utils as img_utils
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow_addons.utils.types import TensorLike

from typing import Optional, Type

_image_so = LazySO("custom_ops/image/_image_ops.so")

tf.no_gradient("Addons>EuclideanDistanceTransform")


def euclidean_dist_transform(
    images: TensorLike,
    dtype: Type[tf.dtypes.DType] = tf.float32,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Applies euclidean distance transform(s) to the image(s).

    Based on [Distance Transforms of Sampled Functions]
    (http://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf).

    Args:
      images: A tensor of shape `(num_images, num_rows, num_columns, num_channels)`
        (NHWC), or `(num_rows, num_columns, num_channels)` (HWC) or
        `(num_rows, num_columns)` (HW).
      dtype: `tf.dtypes.DType` of the output tensor.
      name: The name of the op.

    Returns:
      Image(s) with the type `dtype` and same shape as `images`, with the
      transform applied. If a tensor of all ones is given as input, the
      output tensor will be filled with the max value of the `dtype`.

    Raises:
      TypeError: If `image` is not tf.uint8, or `dtype` is not floating point.
    """

    with tf.name_scope(name or "euclidean_distance_transform"):
        image_or_images = tf.convert_to_tensor(images, name="images")

        if image_or_images.dtype.base_dtype != tf.uint8:
            raise TypeError("Invalid dtype %s. Expected uint8." % image_or_images.dtype)

        images = img_utils.to_4D_image(image_or_images)
        original_ndims = img_utils.get_ndims(image_or_images)

        if dtype not in [tf.float16, tf.float32, tf.float64]:
            raise TypeError("`dtype` must be float16, float32 or float64")

        output = _image_so.ops.addons_euclidean_distance_transform(images, dtype)

        return img_utils.from_4D_image(output, original_ndims)
