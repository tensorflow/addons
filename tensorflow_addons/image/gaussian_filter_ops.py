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


import numpy as np
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

    if kSize % 2 == 0:
        raise ValueError("kSize should be odd")

    gaussianFilter = tf.Variable(tf.zeros(shape=(kSize, kSize), dtype=tf.float64))
    gaussianFilter = findKernel(sigma, kSize, gaussianFilter)

    gaussianKernel = tf.expand_dims(gaussianFilter, axis=2)
    gaussianKernel = tf.expand_dims(gaussianKernel, axis=2)

    conv_ops = tf.nn.convolution(input=img, filters=gaussianKernel, padding="SAME")
    return conv_ops


def findKernel(sigma, kSize, gaussianFilter):
    "This function creates a kernel of size [kSize*kSize]"
    rowFilter = tf.Variable(tf.zeros(kSize, dtype=tf.float64), dtype=tf.float64)
    for i in range(-kSize // 2, kSize // 2 + 1):
        for j in range(-kSize // 2, kSize // 2 + 1):
            rowFilter[j + kSize // 2].assign(
                1
                / (2 * np.pi * (sigma) ** 2)
                * tf.exp(tf.cast(-(i ** 2 + j ** 2) / (2 * sigma ** 2.00), tf.float64))
            )
        gaussianFilter[i + kSize // 2].assign(rowFilter)
    s = tf.math.reduce_sum(gaussianFilter)
    gaussianFilter = tf.math.divide(gaussianFilter, s)
    return gaussianFilter
