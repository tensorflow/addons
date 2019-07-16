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
"""Image util ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_ndims(image):
    return image.get_shape().ndims or tf.rank(image)


@tf.function
def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.

    Args:
      image: 2/3/4D tensor.

    Returns:
      4D tensor with the same type.
    """
    # yapf:disable
    with tf.control_dependencies([
        tf.debugging.assert_rank_in(
            image, [2, 3, 4], message='`image` must be 2/3/4D tensor')
    ]):
        # yapf: enable
        ndims = image.get_shape().ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image


def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    # yapf: disable
    new_shape = tf.concat(
        [tf.ones(shape=left_pad, dtype=tf.int32),
         shape,
         tf.ones(shape=right_pad, dtype=tf.int32)],
        axis=0)
    # yapf: enable
    return tf.reshape(image, new_shape)


@tf.function
def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.

    Args:
      image: 4D tensor.
      ndims: The original rank of the image.

    Returns:
      `ndims`-D tensor with the same type.
    """
    # yapf:disable
    with tf.control_dependencies([
        tf.debugging.assert_rank(
            image, 4, message='`image` must be 4D tensor')
    ]):
        # yapf:enable
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)
