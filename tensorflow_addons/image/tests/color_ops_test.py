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
"""Tests of color ops"""

import pytest
import tensorflow as tf
import numpy as np

from tensorflow_addons.image import color_ops
from PIL import Image, ImageOps, ImageEnhance
from skimage.exposure import equalize_hist

_DTYPES = {
    np.uint8,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
}

_SHAPES = {(5, 5), (5, 5, 1), (5, 5, 3), (4, 5, 5), (4, 5, 5, 1), (4, 5, 5, 3)}


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_equalize_dtype_shape(dtype, shape):
    image = np.ones(shape=shape, dtype=dtype)
    equalized = color_ops.equalize(tf.constant(image)).numpy()
    np.testing.assert_equal(equalized, image)
    assert equalized.dtype == image.dtype


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_equalize_with_PIL():
    np.random.seed(0)
    image = np.random.randint(low=0, high=255, size=(4, 3, 3, 3), dtype=np.uint8)
    equalized = np.stack([ImageOps.equalize(Image.fromarray(i)) for i in image])
    np.testing.assert_equal(color_ops.equalize(tf.constant(image)).numpy(), equalized)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("bins", [256, 65536])
def test_equalize_with_skimage(bins):
    np.random.seed(0)
    image = np.random.randint(low=0, high=256, size=(5, 5, 3, 3), dtype=np.uint8)
    equalized = tf.cast(color_ops.equalize(tf.constant(image), bins), np.float16)
    equalized = tf.math.divide(equalized, 255)
    skiequalized = equalize_hist(image, bins)
    assert (
        tf.reduce_mean(
            tf.math.abs(tf.reshape(tf.math.subtract(equalized, skiequalized), [-1]))
        )
        <= 0.015
    ), "Greater than 0.015 mean absolute difference in arrays"
    # Test asserts a tolerance of at least 0.015 mean difference between the two
    # arrays due to the difference in implementation between PIL and skimage.
    # PIL's and skimage's equalized arrays can have a mean difference ranging
    # from 0.006-0.036 according showcase done in
    # https://github.com/tensorflow/addons/pull/2381


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_sharpness_dtype_shape(dtype, shape):
    image = np.ones(shape=shape, dtype=dtype)
    sharp = color_ops.sharpness(tf.constant(image), 0).numpy()
    np.testing.assert_equal(sharp, image)
    assert sharp.dtype == image.dtype


@pytest.mark.parametrize("factor", [0, 0.25, 0.5, 0.75, 1])
def test_sharpness_with_PIL(factor):
    np.random.seed(0)
    image = np.random.randint(low=0, high=255, size=(10, 5, 5, 3), dtype=np.uint8)
    sharpened = np.stack(
        [ImageEnhance.Sharpness(Image.fromarray(i)).enhance(factor) for i in image]
    )
    np.testing.assert_allclose(
        color_ops.sharpness(tf.constant(image), factor).numpy(), sharpened, atol=1
    )
