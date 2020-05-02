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
"RGB to Grayscale op"

from tensorflow_addons.utils.types import FloatTensorLike
from tensorflow_addons.image import utils as img_utils
import tensorflow as tf

from typing import Optional


def rgb_to_grayscale(
    image: FloatTensorLike, name: Optional[str] = None,
) -> FloatTensorLike:
    """Perform RGB to Grayscale conversion of image(s).

    Args:
      image: A 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
        The number of channels must be equal to 3.
      name: A name for this operation (optional).
    """

    with tf.name_scope(name or "rgb_to_grayscale"):
        image = tf.cast(image, tf.float32)
        original_ndims = img_utils.get_ndims(image)
        image = img_utils.to_4D_image(image)
        channels = tf.shape(image)[3]
        if channels != 3:
            raise ValueError("The image must be in RGB format")
        grayscale_output = tf.image.rgb_to_grayscale(image)
        grayscale_output = img_utils.from_4D_image(grayscale_output, original_ndims)
        return grayscale_output
