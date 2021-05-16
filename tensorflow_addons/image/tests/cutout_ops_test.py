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

import pytest
import tensorflow as tf
import numpy as np
from tensorflow_addons.image.cutout_ops import cutout, random_cutout
from tensorflow_addons.image.utils import to_4D_image


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.uint8])
def test_different_dtypes(dtype):
    test_image = tf.ones([1, 40, 40, 1], dtype=dtype)
    result_image = cutout(test_image, 4, [2, 2])
    cutout_area = tf.zeros([4, 4], dtype=dtype)
    cutout_area = tf.pad(cutout_area, ((0, 36), (0, 36)), constant_values=1)
    expect_image = to_4D_image(cutout_area)
    np.testing.assert_allclose(result_image, expect_image)
    assert result_image.dtype == dtype


def test_different_channels():
    for channel in [0, 1, 3, 4]:
        test_image = tf.ones([1, 40, 40, channel], dtype=np.uint8)
        cutout_area = tf.zeros([4, 4], dtype=np.uint8)
        cutout_area = tf.pad(cutout_area, ((0, 36), (0, 36)), constant_values=1)
        expect_image = to_4D_image(cutout_area)
        expect_image = tf.tile(expect_image, [1, 1, 1, channel])
        result_image = random_cutout(test_image, 20, seed=1234)
        np.testing.assert_allclose(tf.shape(result_image), tf.shape(expect_image))


def test_batch_size():
    test_image = tf.random.uniform([10, 40, 40, 1], dtype=np.float32, seed=1234)
    result_image = random_cutout(test_image, 20, seed=1234)
    np.testing.assert_allclose(tf.shape(result_image), [10, 40, 40, 1])
    means = np.mean(result_image, axis=(1, 2, 3))
    np.testing.assert_allclose(len(set(means)), 10)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_with_tf_function():
    test_image = tf.ones([1, 40, 40, 1], dtype=tf.uint8)
    result_image = tf.function(random_cutout)(test_image, 2)
    cutout_area = tf.zeros([4, 4], dtype=tf.uint8)
    cutout_area = tf.pad(cutout_area, ((0, 36), (0, 36)), constant_values=1)
    expect_image = to_4D_image(cutout_area)
    np.testing.assert_equal(result_image.shape, expect_image.shape)


def test_mask_applied():
    test_image = tf.ones([10, 40, 40, 1], dtype=np.uint8)
    result_image = random_cutout(test_image, 20, seed=1234)
    total_expected_masked_count = 20 * 20 * test_image.shape[0]

    np.testing.assert_equal(
        np.sum(result_image) + total_expected_masked_count, np.sum(test_image)
    )


def test_mask_larger_than_image():
    test_image = tf.ones([10, 40, 40, 1], dtype=np.uint8)
    result_image = random_cutout(test_image, 60, seed=1234)
    np.testing.assert_equal(np.sum(result_image), 0)


def test_keras_layer():
    inputs = tf.keras.Input(shape=(40, 40, 1), dtype=tf.uint8)
    outputs = tf.keras.layers.Lambda(lambda x: cutout(x, 4, [2, 2]))(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    x = tf.ones([10, 40, 40, 1], dtype=np.uint8)
    np.testing.assert_equal(cutout(x, 4, [2, 2]).numpy(), model(x))


def test_invalid_mask_size():
    with pytest.raises(tf.errors.InvalidArgumentError, match="mask_size should be"):
        x = tf.ones([10, 40, 40, 1], dtype=np.uint8)
        cutout(x, 3, [2, 2]).numpy()
