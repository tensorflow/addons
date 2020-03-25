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
"""Color Ops"""

import tensorflow as tf

from tensorflow_addons.utils.types import TensorLike


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(image, channel=None):
        """Scale the data in the channel to implement equalize."""
        image_dtype = image.dtype
        image = tf.cast(image[:, :, channel], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(image, [0, 255], nbins=256, dtype=image.dtype)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(
            tf.equal(step, 0),
            lambda: image,
            lambda: tf.gather(build_lut(histo, step), image),
        )

        return tf.cast(result, image_dtype)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    if tf.rank(image) == 2:
        image = tf.expand_dims(image, axis=2)
        image = scale_channel(image, 0)
        return tf.squeeze(image)

    if image.shape[2] == 1:
        image = scale_channel(image, 0)
        return tf.expand_dims(image, 2)

    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image
