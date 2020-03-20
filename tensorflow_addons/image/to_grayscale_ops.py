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
""" method to convert the color images into grayscale
by keeping the channel same"""
import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike


def to_grayscale(image: TensorLike, keep_channels: bool = True) -> TensorLike:
    """ Method to convert the color image into grayscale
    by keeping the channels same.

    Args:
      image: color image to convert into grayscale
      keep_channels: boolean parameter for channels
    Returns:
      Image"""
    image = tf.image.rgb_to_grayscale(image)
    if keep_channels:
        image = tf.tile(image, [1, 1, 3])
    return image
