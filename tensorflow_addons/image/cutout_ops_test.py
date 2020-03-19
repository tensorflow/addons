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
"""Tests for cutout."""

import sys

import pytest
import tensorflow as tf
import numpy as np
from absl.testing import parameterized
from tensorflow_addons.image.cutout_ops import cutout, random_cutout
from tensorflow_addons.image.utils import from_4D_image, to_4D_image


@parameterized.named_parameters(
    ("float16", np.float16), ("float32", np.float32), ("uint8", np.uint8)
)
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_different_dtypes(dtype):
    test_image = tf.ones([1, 40, 40, 1], dtype=dtype)
    result_image = cutout(test_image, 2, [2, 2])
    cutout_area = tf.zeros([4, 4], dtype=dtype)
    cutout_area = tf.pad(cutout_area, ((0, 36), (0, 36)), constant_values=1)
    expect_image = to_4D_image(cutout_area)
    np.testing.assert_allclose(result_image, expect_image)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_different_channels():
    test_image = tf.ones([1, 40, 40, 1], dtype=np.uint8)
    cutout_area = tf.zeros([4, 4], dtype=np.uint8)
    cutout_area = tf.pad(cutout_area, ((0, 36), (0, 36)), constant_values=1)
    expect_image = to_4D_image(cutout_area)
    for channel in [0, 1, 3, 4]:
        result_image = random_cutout(test_image, 20, seed=1234)
        np.testing.assert_allclose(tf.shape(result_image), tf.shape(expect_image))


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_different_ranks():
    test_image_4d = tf.ones([1, 40, 40, 1], dtype=np.uint8)
    cutout_area = tf.zeros([4, 4], dtype=np.uint8)
    cutout_area = tf.pad(cutout_area, ((0, 36), (0, 36)), constant_values=1)
    expect_image_4d = to_4D_image(cutout_area)

    test_image_2d = from_4D_image(test_image_4d, 2)
    expect_image_2d = from_4D_image(expect_image_4d, 2)
    result_image_2d = random_cutout(test_image_2d, 20, seed=1234)
    np.testing.assert_allclose(tf.shape(result_image_2d), tf.shape(expect_image_2d))

    result_image_4d = random_cutout(test_image_4d, 20, seed=1234)
    np.testing.assert_allclose(tf.shape(result_image_4d), tf.shape(expect_image_4d))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
