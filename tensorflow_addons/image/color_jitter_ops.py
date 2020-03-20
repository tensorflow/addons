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
""" This method is used to distorts the color of the image """
import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike


def color_jitter(
    image: TensorLike, strength: float, random_order: bool = True
) -> TensorLike:
    """Distorts the color of the image.
  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation <= 2.0.
    random_order: A bool, specifying whether to randomize the jittering order.
  Returns:
    The distorted image tensor.
  """
    image = tf.cast(image, dtype=tf.dtypes.float32)
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    if random_order:
        return color_jitter_rand(image, brightness, contrast, saturation, hue)
    else:
        return color_jitter_nonrand(image, brightness, contrast, saturation, hue)


def color_jitter_nonrand(
    image: TensorLike,
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    hue: float = 0,
) -> TensorLike:
    """Distorts the color of the image (jittering order is fixed).
  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
  Returns:
    The distorted image tensor.
  """
    with tf.name_scope("distort_color"):

        def apply_transform(i, x, brightness, contrast, saturation, hue):
            """Apply the i-th transformation."""
            if brightness != 0 and i == 0:
                x = tf.image.random_brightness(x, max_delta=brightness)
            elif contrast != 0 and i == 1:
                x = tf.image.random_contrast(x, lower=1 - contrast, upper=1 + contrast)
            elif saturation != 0 and i == 2:
                x = tf.image.random_saturation(
                    x, lower=1 - saturation, upper=1 + saturation
                )
            elif hue != 0:
                x = tf.image.random_hue(x, max_delta=hue)
            return x

        for i in range(4):
            image = apply_transform(i, image, brightness, contrast, saturation, hue)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def color_jitter_rand(
    image: TensorLike,
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    hue: float = 0,
) -> TensorLike:
    """Distorts the color of the image (jittering order is random).
  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
  Returns:
    The distorted image tensor.
  """
    with tf.name_scope("distort_color"):

        def apply_transform(i, x):
            """Apply the i-th transformation."""

            def brightness_foo():
                if brightness == 0:
                    return x
                else:
                    return tf.image.random_brightness(x, max_delta=brightness)

            def contrast_foo():
                if contrast == 0:
                    return x
                else:
                    return tf.image.random_contrast(
                        x, lower=tf.math.abs(1 - contrast), upper=1 + contrast
                    )

            def saturation_foo():
                if saturation == 0:
                    return x
                else:
                    return tf.image.random_saturation(
                        x, lower=tf.math.abs(1 - saturation), upper=1 + saturation
                    )

            def hue_foo():
                if hue == 0:
                    return x
                else:
                    return tf.image.random_hue(x, max_delta=hue)

            x = tf.cond(
                tf.less(i, 2),
                lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo),
            )
            return x

        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            image = apply_transform(perm[i], image)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return tf.cast(image, dtype=tf.uint8)
