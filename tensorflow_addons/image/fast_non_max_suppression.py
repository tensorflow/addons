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
from tensorflow_addons.utils.resource_loader import get_path_to_datafile

_fast_non_max_suppression_ops = tf.load_op_library(
    get_path_to_datafile("custom_ops/image/_fast_non_max_suppression_ops.so"))


@tf.function
def fast_non_max_suppression(image,
                             boxes,
                             scores,
                             max_output_size,
                             iou_threshold=0.5,
                             score_threshold=float('-inf'),
                             top_k=200,
                             name=None):
    """Adjust hue, saturation, value of an RGB image in YIQ color space.

    This is a convenience method that converts an RGB image to float
    representation, converts it to YIQ, rotates the color around the
    Y channel by delta_hue in radians, scales the chrominance channels
    (I, Q) by scale_saturation, scales all channels (Y, I, Q) by scale_value,
    converts back to RGB, and then back to the original data type.

    `image` is an RGB image. The image hue is adjusted by converting the
    image to YIQ, rotating around the luminance channel (Y) by
    `delta_hue` in radians, multiplying the chrominance channels (I, Q) by
    `scale_saturation`, and multiplying all channels (Y, I, Q) by
    `scale_value`. The image is then converted back to RGB.

    Args:
      image: RGB image or images. Size of the last dimension must be 3.
      delta_hue: float, the hue rotation amount, in radians.
      scale_saturation: float, factor to multiply the saturation by.
      scale_value: float, factor to multiply the value by.
      name: A name for this operation (optional).

    Returns:
      Adjusted image(s), same shape and dtype as `image`.
    """
    with tf.name_scope(name or "fast_non_max_suppression"):

        idx = tf.argsort(scores, direction='DESCENDING')[:, :top_k]
        scores = tf.sort(scores, direction='DESCENDING')[:, :top_k]
        boxes = tf.gather(boxes, idx, axis=1, batch_dims=1)

        selected_indices = _fast_non_max_suppression_ops.addons_fast_non_max_suppression(
            boxes, scores, iou_threshold, score_threshold)
        return tf.boolean_mask(selected_indices * idx, selected_indices)[:,:max_output_size]
