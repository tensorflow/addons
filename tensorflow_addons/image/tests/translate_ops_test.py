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
"""Tests for translate ops."""

import numpy as np
import pytest
import scipy
import tensorflow as tf

from PIL import Image

from tensorflow_addons.image import translate_ops
from tensorflow_addons.utils import test_utils

_DTYPES = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.int64,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_translate(dtype):
    image = tf.constant(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=dtype
    )
    translation = tf.constant([-1, -1], dtype=tf.float32)
    image_translated = translate_ops.translate(image, translation)
    np.testing.assert_equal(
        image_translated.numpy(),
        [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_translations_to_projective_transforms():
    translation = tf.constant([-1, -1], dtype=tf.float32)
    transform = translate_ops.translations_to_projective_transforms(translation)
    np.testing.assert_equal(transform.numpy(), [[1, 0, 1, 0, 1, 1, 0, 0]])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_translate_xy():
    image = np.random.randint(low=0, high=255, size=(4, 4, 3), dtype=np.uint8)
    translate = np.random.randint(low=0, high=4, size=(2,), dtype=np.uint8)
    translate = tf.constant(translate)
    color = tf.constant([255, 0, 255], tf.dtypes.uint8)

    tf_image = tf.constant(image)
    pil_image = Image.fromarray(image)

    translated = translate_ops.translate_xy(
        image=tf_image, translate_to=tf.constant(translate), replace=color
    )
    expected = pil_image.rotate(
        angle=0,
        resample=Image.NEAREST,
        translate=tuple(translate.numpy()),
        fillcolor=tuple(color.numpy()),
    )

    np.testing.assert_equal(translated.numpy(), expected)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES - {tf.dtypes.float16})
def test_translate_xy_scalar_replace(dtype):
    image = np.random.randint(low=0, high=128, size=(4, 4, 3)).astype(
        dtype.as_numpy_dtype
    )
    translate_to = np.random.randint(low=0, high=4, size=(2,))
    result = translate_ops.translate_xy(
        image=image, translate_to=translate_to, replace=1
    )
    expected = scipy.ndimage.shift(
        input=image,
        shift=(translate_to[1], translate_to[0], 0),
        order=0,
        mode="constant",
        cval=1,
    )
    test_utils.assert_allclose_according_to_type(result.numpy(), expected)
