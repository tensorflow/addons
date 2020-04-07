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
# =============================================================================
"""GaussuanBlur Op"""


import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike


def gaussian_blur(
    img: FloatTensorLike, sigma: tf.float64, kSize: tf.float64
) -> FloatTensorLike:
    """
    This function is responsible for having Gaussian Blur. It takes the image as input, computes a gaussian-kernel
    which follows normal distribution then convolves the image with the kernel.Presently it works only on
    grayscale images.
    Args:
    img: A tensor of shape
        (batch_size, height, width, channels)
        (NHWC), (batch_size, channels, height, width)(NCHW).
    sigma:A constant of type float64. It is the standard deviation of the normal distribution.
          The more the sigma, the more the blurring effect.
          G(x,y)=1/(2*3.14*sigma**2)e^((x**2+y**2)/2sigma**2)
    kSize:It is the kernel-size for the Gaussian Kernel. kSize should be odd.
          A kernel of size [kSize*kSize] is generated.
    """

    if sigma == 0:
        raise ValueError("Sigma should not be zero")

    gaussian_filter_x = find_kernel(sigma, kSize, axis="x")
    gaussian_filter_y = find_kernel(sigma, kSize, axis="y")

    conv_ops_x = tf.nn.convolution(input=img, filters=gaussian_filter_x, padding="SAME")
    conv_ops = tf.nn.convolution(
        input=conv_ops_x, filters=gaussian_filter_y, padding="SAME"
    )
    return conv_ops


def find_kernel(sigma, kSize, axis="x"):
    "This function creates a kernel of size [kSize]"
    x = tf.range(-kSize // 2 + 1, kSize // 2 + 1)
    x = tf.math.square(x, tf.float64)
    a = tf.cast(tf.exp(-(x) / (2 * (sigma ** 2))), tf.float64)
    a = a / tf.math.reduce_sum(a)
    if axis == "y":
        a = tf.reshape(a, [kSize, 1, 1, 1])
    else:
        a = tf.reshape(a, [1, kSize, 1, 1])
    return a
