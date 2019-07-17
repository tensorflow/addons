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
"""Image translate ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.image.transform_ops import transform


@tf.function
def translations_to_projective_transforms(translations, name=None):
    """Returns projective transform(s) for the given translation(s).

    Args:
        translations: A 2-element list representing [dx, dy] or a matrix of
            2-element lists representing [dx, dy] to translate for each image
            (for a batch of images). The rank must be statically known
            (the shape is not `TensorShape(None)`).
        name: The name of the op.
    Returns:
        A tensor of shape (num_images, 8) projective transforms which can be
        given to `tfa.image.transform`.
    """
    with tf.name_scope(name or "translations_to_projective_transforms"):
        translation_or_translations = tf.convert_to_tensor(
            translations, name="translations", dtype=tf.dtypes.float32)
        if translation_or_translations.get_shape().ndims is None:
            raise TypeError(
                "translation_or_translations rank must be statically known")
        elif len(translation_or_translations.get_shape()) == 1:
            translations = translation_or_translations[None]
        elif len(translation_or_translations.get_shape()) == 2:
            translations = translation_or_translations
        else:
            raise TypeError("Translations should have rank 1 or 2.")
        num_translations = tf.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        return tf.concat(
            values=[
                tf.ones((num_translations, 1), tf.dtypes.float32),
                tf.zeros((num_translations, 1), tf.dtypes.float32),
                -translations[:, 0, None],
                tf.zeros((num_translations, 1), tf.dtypes.float32),
                tf.ones((num_translations, 1), tf.dtypes.float32),
                -translations[:, 1, None],
                tf.zeros((num_translations, 2), tf.dtypes.float32),
            ],
            axis=1)


@tf.function
def translate(images, translations, interpolation="NEAREST", name=None):
    """Translate image(s) by the passed vectors(s).

    Args:
      images: A tensor of shape
          (num_images, num_rows, num_columns, num_channels) (NHWC),
          (num_rows, num_columns, num_channels) (HWC), or
          (num_rows, num_columns) (HW). The rank must be statically known (the
          shape is not `TensorShape(None)`).
      translations: A vector representing [dx, dy] or (if images has rank 4)
          a matrix of length num_images, with a [dx, dy] vector for each image
          in the batch.
      interpolation: Interpolation mode. Supported values: "NEAREST",
          "BILINEAR".
      name: The name of the op.
    Returns:
      Image(s) with the same type and shape as `images`, translated by the
      given vector(s). Empty space due to the translation will be filled with
      zeros.
    Raises:
      TypeError: If `images` is an invalid type.
    """
    with tf.name_scope(name or "translate"):
        return transform(
            images,
            translations_to_projective_transforms(translations),
            interpolation=interpolation)
