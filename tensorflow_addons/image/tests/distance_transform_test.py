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
"""Tests for distance transform ops."""

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.image import distance_transform as dist_ops
from tensorflow_addons.utils import test_utils


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32, tf.float64])
def test_single_binary_image(dtype):
    image = [
        [[1], [1], [1], [1], [1]],
        [[1], [1], [1], [1], [1]],
        [[0], [1], [0], [1], [0]],
        [[1], [0], [1], [0], [1]],
        [[0], [1], [0], [1], [0]],
    ]
    expected_output = np.array(
        [
            2,
            2.23606801,
            2,
            2.23606801,
            2,
            1,
            1.41421354,
            1,
            1.41421354,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
        ]
    )
    image = tf.constant(image, dtype=tf.uint8)

    output = dist_ops.euclidean_dist_transform(image, dtype=dtype)
    output_flat = tf.reshape(output, [-1])

    assert output.dtype == dtype
    assert output.shape == [5, 5, 1]
    test_utils.assert_allclose_according_to_type(output_flat, expected_output)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32, tf.float64])
def test_batch_binary_images(dtype):
    batch_size = 3
    image = [
        [[0], [0], [0], [0], [0]],
        [[0], [1], [1], [1], [0]],
        [[0], [1], [1], [1], [0]],
        [[0], [1], [1], [1], [0]],
        [[0], [0], [0], [0], [0]],
    ]
    expected_output = np.array(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        * batch_size
    )
    images = tf.constant([image] * batch_size, dtype=tf.uint8)

    output = dist_ops.euclidean_dist_transform(images, dtype=dtype)
    output_flat = tf.reshape(output, [-1])

    assert output.shape == [batch_size, 5, 5, 1]
    test_utils.assert_allclose_according_to_type(output_flat, expected_output)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.uint8, tf.int32, tf.int64])
def test_image_with_invalid_dtype(dtype):
    image = [
        [[1], [1], [1], [1], [1]],
        [[1], [1], [1], [1], [1]],
        [[0], [1], [0], [1], [0]],
        [[1], [0], [1], [0], [1]],
        [[0], [1], [0], [1], [0]],
    ]
    image = tf.constant(image, dtype=tf.uint8)

    with pytest.raises(TypeError, match="`dtype` must be float16, float32 or float64"):
        _ = dist_ops.euclidean_dist_transform(image, dtype=dtype)


@pytest.mark.with_device(["cpu", "gpu"])
def test_all_zeros():
    image = tf.zeros([10, 10], tf.uint8)
    expected_output = np.zeros([10, 10])

    for output_dtype in [tf.float16, tf.float32, tf.float64]:
        output = dist_ops.euclidean_dist_transform(image, dtype=output_dtype)
        np.testing.assert_allclose(output, expected_output)


@pytest.mark.with_device(["cpu", "gpu"])
def test_all_ones():
    image = tf.ones([10, 10, 1], tf.uint8)
    output = dist_ops.euclidean_dist_transform(image)
    expected_output = np.full([10, 10, 1], tf.math.sqrt(tf.float32.max))
    np.testing.assert_allclose(output, expected_output)


@pytest.mark.with_device(["cpu", "gpu"])
def test_multi_channels():
    channels = 3
    batch_size = 2048
    image = [
        [[0], [0], [0], [0], [0]],
        [[0], [1], [1], [1], [0]],
        [[0], [1], [1], [1], [0]],
        [[0], [1], [1], [1], [0]],
        [[0], [0], [0], [0], [0]],
    ]
    expected_output = np.tile(
        np.expand_dims(
            np.array(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    2,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
            axis=-1,
        ),
        [batch_size, 3],
    )
    image = np.tile(image, [1, 1, channels])
    images = tf.constant([image] * batch_size, dtype=tf.uint8)

    output = dist_ops.euclidean_dist_transform(images, dtype=tf.float32)
    output_flat = tf.reshape(output, [-1, 3])
    assert output.shape == [batch_size, 5, 5, channels]
    test_utils.assert_allclose_according_to_type(output_flat, expected_output)
