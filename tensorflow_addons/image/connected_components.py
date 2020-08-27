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
"""Connected Components."""

import tensorflow as tf

from tensorflow_addons.utils import types
from tensorflow_addons.utils.resource_loader import LazySO

from typing import Optional, Text

_image_so = LazySO("custom_ops/image/_image_ops.so")


@tf.function
def connected_components(
    images: types.TensorLike, name: Optional[Text] = None
) -> tf.Tensor:
    """Labels the connected components in a batch of images.

    A component is a set of pixels in a single input image, which are
    all adjacent and all have the same non-zero value. The components
    using a squared connectivity of one (all equal entries are joined with
    their neighbors above,below, left, and right). Components across all
    images have consecutive ids 1 through n.
    Components are labeled according to the first pixel of the
    component appearing in row-major order (lexicographic order by
    image_index_in_batch, row, col).
    Zero entries all have an output id of 0.
    This op is equivalent with `scipy.ndimage.measurements.label`
    on a 2D array with the default structuring element
    (which is the connectivity used here).

    Args:
      images: A 2D (H, W) or 3D (N, H, W) `Tensor` of image (integer,
      floating point and boolean types are supported).
      name: The name of the op.

    Returns:
      Components with the same shape as `images`.
      entries that evaluate to False (e.g. 0/0.0f, False) in `images` have
      value 0, and all other entries map to a component id > 0.

    Raises:
      TypeError: if `images` is not 2D or 3D.
    """
    with tf.name_scope(name or "connected_components"):
        image_or_images = tf.convert_to_tensor(images, name="images")
        if len(image_or_images.get_shape()) == 2:
            images = image_or_images[None, :, :]
        elif len(image_or_images.get_shape()) == 3:
            images = image_or_images
        else:
            raise TypeError(
                "images should have rank 2 (HW) or 3 (NHW). Static shape is %s"
                % image_or_images.get_shape()
            )
        components = _image_so.ops.addons_image_connected_components(images)

        # TODO(ringwalt): Component id renaming should be done in the op,
        # to avoid constructing multiple additional large tensors.
        components_flat = tf.reshape(components, [-1])
        unique_ids, id_index = tf.unique(components_flat)
        id_is_zero = tf.where(tf.equal(unique_ids, 0))[:, 0]
        # Map each nonzero id to consecutive values.
        nonzero_consecutive_ids = (
            tf.range(tf.shape(unique_ids)[0] - tf.shape(id_is_zero)[0]) + 1
        )

        def no_zero():
            # No need to insert a zero into the ids.
            return nonzero_consecutive_ids

        def has_zero():
            # Insert a zero in the consecutive ids
            # where zero appears in unique_ids.
            # id_is_zero has length 1.
            zero_id_ind = tf.cast(id_is_zero[0], tf.int32)
            ids_before = nonzero_consecutive_ids[:zero_id_ind]
            ids_after = nonzero_consecutive_ids[zero_id_ind:]
            return tf.concat([ids_before, [0], ids_after], axis=0)

        new_ids = tf.cond(tf.equal(tf.shape(id_is_zero)[0], 0), no_zero, has_zero)
        components = tf.reshape(tf.gather(new_ids, id_index), tf.shape(components))
        if len(image_or_images.get_shape()) == 2:
            return components[0, :, :]
        else:
            return components
