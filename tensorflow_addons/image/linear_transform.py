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
"""Op for linear transformation."""
import tensorflow as tf
from tensorflow_addons.image import utils as img_utils
from tensorflow_addons.utils.types import FloatTensorLike
from typing import Optional


def linear_transform(
    image: FloatTensorLike,
    a: FloatTensorLike = 1,
    b: FloatTensorLike = 1,
    name: Optional[str] = None,
):
    """Linear transformation on an image.

    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      a: It is an integer or float or double representing slope of the line.
      b: It is an integer or float or double representing y-intercept.
      name: A name for the operation.
    """
    with tf.name_scope(name or "gaussian_filter2d"):
        image = tf.convert_to_tensor(image, name="image")
        original_ndims = img_utils.get_ndims(image)
        image = tf.cast(image, tf.float32)
        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)
        image = img_utils.to_4D_image(image)
        output = a * image + b
        output = img_utils.from_4D_image(output, original_ndims)
        return output
