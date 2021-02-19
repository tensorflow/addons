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
"""Python layer for distort_image_ops."""

from typing import Optional
import warnings

import tensorflow as tf

from tensorflow_addons import options
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow_addons.utils.types import Number, TensorLike

_distort_image_so = LazySO("custom_ops/image/_distort_image_ops.so")


def random_hsv_in_yiq(
    image: TensorLike,
    max_delta_hue: Number = 0,
    lower_saturation: Number = 1,
    upper_saturation: Number = 1,
    lower_value: Number = 1,
    upper_value: Number = 1,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Adjust hue, saturation, value of an RGB image randomly in YIQ color space.

    Equivalent to `adjust_yiq_hsv()` but uses a `delta_h` randomly
    picked in the interval `[-max_delta_hue, max_delta_hue]`, a
    `scale_saturation` randomly picked in the interval
    `[lower_saturation, upper_saturation]`, and a `scale_value`
    randomly picked in the interval `[lower_saturation, upper_saturation]`.

    Args:
      image: RGB image or images. Size of the last dimension must be 3.
      max_delta_hue: `float`. Maximum value for the random delta_hue. Passing 0
        disables adjusting hue.
      lower_saturation: `float`. Lower bound for the random scale_saturation.
      upper_saturation: `float`. Upper bound for the random scale_saturation.
      lower_value: `float`. Lower bound for the random scale_value.
      upper_value: `float`. Upper bound for the random scale_value.
      seed: An operation-specific seed. It will be used in conjunction
        with the graph-level seed to determine the real seeds that will be
        used in this operation. Please see the documentation of
        set_random_seed for its interaction with the graph-level random seed.
      name: A name for this operation (optional).

    Returns:
      3-D float `Tensor` of shape `[height, width, channels]`.

    Raises:
      ValueError: if `max_delta`, `lower_saturation`, `upper_saturation`,
        `lower_value`, or `upper_value` is invalid.
    """
    if max_delta_hue < 0:
        raise ValueError("max_delta must be non-negative.")

    if lower_saturation < 0:
        raise ValueError("lower_saturation must be non-negative.")

    if lower_value < 0:
        raise ValueError("lower_value must be non-negative.")

    if lower_saturation > upper_saturation:
        raise ValueError(
            "lower_saturation must be not greater than " "upper_saturation."
        )

    if lower_value > upper_value:
        raise ValueError("lower_value must be not greater than upper_value.")

    with tf.name_scope(name or "random_hsv_in_yiq") as scope:
        if max_delta_hue == 0:
            delta_hue = 0
        else:
            delta_hue = tf.random.uniform([], -max_delta_hue, max_delta_hue, seed=seed)
        if lower_saturation == upper_saturation:
            scale_saturation = lower_saturation
        else:
            scale_saturation = tf.random.uniform(
                [], lower_saturation, upper_saturation, seed=seed
            )
        if lower_value == upper_value:
            scale_value = lower_value
        else:
            scale_value = tf.random.uniform([], lower_value, upper_value, seed=seed)
        return adjust_hsv_in_yiq(
            image, delta_hue, scale_saturation, scale_value, name=scope
        )


def _adjust_hsv_in_yiq(
    image,
    delta_hue,
    scale_saturation,
    scale_value,
):
    if image.shape.rank is not None and image.shape.rank < 3:
        raise ValueError("input must be at least 3-D.")
    if image.shape[-1] is not None and image.shape[-1] != 3:
        raise ValueError(
            "input must have 3 channels but instead has {}.".format(image.shape[-1])
        )
    # Construct hsv linear transformation matrix in YIQ space.
    # https://beesbuzz.biz/code/hsv_color_transforms.php
    yiq = tf.constant(
        [[0.299, 0.596, 0.211], [0.587, -0.274, -0.523], [0.114, -0.322, 0.312]],
        dtype=tf.float32,
    )
    yiq_inverse = tf.constant(
        [
            [1.0, 1.0, 1.0],
            [0.95617069, -0.2726886, -1.103744],
            [0.62143257, -0.64681324, 1.70062309],
        ],
        dtype=tf.float32,
    )
    vsu = scale_value * scale_saturation * tf.math.cos(delta_hue)
    vsw = scale_value * scale_saturation * tf.math.sin(delta_hue)
    hsv_transform = tf.convert_to_tensor(
        [[scale_value, 0, 0], [0, vsu, vsw], [0, -vsw, vsu]], dtype=tf.float32
    )
    transform_matrix = yiq @ hsv_transform @ yiq_inverse

    image = image @ transform_matrix
    return image


def adjust_hsv_in_yiq(
    image: TensorLike,
    delta_hue: Number = 0,
    scale_saturation: Number = 1,
    scale_value: Number = 1,
    name: Optional[str] = None,
) -> tf.Tensor:
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
      delta_hue: `float`, the hue rotation amount, in radians.
      scale_saturation: `float`, factor to multiply the saturation by.
      scale_value: `float`, factor to multiply the value by.
      name: A name for this operation (optional).

    Returns:
      Adjusted image(s), same shape and dtype as `image`.
    """
    with tf.name_scope(name or "adjust_hsv_in_yiq"):
        image = tf.convert_to_tensor(image, name="image")
        delta_hue = tf.cast(delta_hue, dtype=tf.float32, name="delta_hue")
        scale_saturation = tf.cast(
            scale_saturation, dtype=tf.float32, name="scale_saturation"
        )
        scale_value = tf.cast(scale_value, dtype=tf.float32, name="scale_value")

        # Remember original dtype to so we can convert back if needed
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)

        if not options.TF_ADDONS_PY_OPS:
            warnings.warn(
                "C++/CUDA kernel of `adjust_hsv_in_yiq` will be removed in Addons `0.13`.",
                DeprecationWarning,
            )
            try:
                image = _distort_image_so.ops.addons_adjust_hsv_in_yiq(
                    image, delta_hue, scale_saturation, scale_value
                )
            except tf.errors.NotFoundError:
                options.warn_fallback("adjust_hsv_in_yiq")
                image = _adjust_hsv_in_yiq(
                    image, delta_hue, scale_saturation, scale_value
                )
        else:
            image = _adjust_hsv_in_yiq(image, delta_hue, scale_saturation, scale_value)

        return tf.image.convert_image_dtype(image, orig_dtype)
