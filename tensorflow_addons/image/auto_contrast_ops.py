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
""" Maximize (normalize) image contrast.
This function calculates a histogram of the input image,
removes cutoff percent of the lightest and darkest pixels
 from the histogram, and remaps the image so
that the darkest pixel becomes black (0),
and the lightest becomes white (255). """

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike


def autocontrast(image: TensorLike) -> TensorLike:
    """Implements Autocontrast function from PIL using TF ops.
  Args:
    image: A 3D uint8 tensor.
  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

    def scale_channel(image: TensorLike) -> TensorLike:
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), dtype=tf.float32)
        hi = tf.cast(tf.reduce_max(image), dtype=tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im: TensorLike) -> TensorLike:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, dtype=tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image
