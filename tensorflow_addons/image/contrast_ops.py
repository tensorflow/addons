# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf

def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.

    Args:
      image: A 3D uint8 tensor.

    Returns:
      The image after it has had autocontrast applied to it and will be of type
      uint8.
    """
    dtype=image.dtype
    interval=dtype.max - dtype.min
    def scale_channel(image_channel):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.reduce_min(image_channel)
        hi = tf.reduce_max(image_channel)

        if hi > lo:
            image_channel=tf.cast(image_channel,tf.float32)
            image_channel=image_channel*(interval/(hi - lo))
            image_channel=tf.cast(image_channel, dtype)
        return image_channel

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.

    ss=tf.unstack(image,axis=-1)
    for i in range(len(ss)):
        ss[i]=scale_channel(ss[i])
    image = tf.stack(ss, -1)

    return image


