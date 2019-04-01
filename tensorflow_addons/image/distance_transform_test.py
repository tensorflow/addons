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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.image import distance_transform as distance_tranform_ops
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class DistanceOpsTest(tf.test.TestCase):
    def test_single_binary_image(self):
        # yapf: disable
        image = [[[1], [1], [1], [1], [1]],
                 [[1], [1], [1], [1], [1]],
                 [[0], [1], [0], [1], [0]],
                 [[1], [0], [1], [0], [1]],
                 [[0], [1], [0], [1], [0]]]
        # pylint: disable=bad-whitespace
        expected_output = np.array([
            2, 2.23606801, 2, 2.23606801, 2,
            1, 1.41421354, 1, 1.41421354, 1,
            0, 1,          0, 1,          0,
            1, 0,          1, 0,          1,
            0, 1,          0, 1,          0])
        # yapf: enable
        image = tf.constant(image, dtype=tf.uint8)

        for output_dtype in [tf.float16, tf.float32, tf.float64]:
            output = distance_tranform_ops.euclidean_dist_transform(
                image, dtype=output_dtype)
            output_flat = tf.reshape(output, [-1])

            with self.subTest(output_dtype=output_dtype):
                self.assertEqual(output.dtype, output_dtype)
                self.assertEqual(output.shape, [5, 5, 1])
                self.assertAllCloseAccordingToType(output_flat,
                                                   expected_output)

    def test_batch_binary_images(self):
        batch_size = 3
        # yapf: disable
        image = [[[0], [0], [0], [0], [0]],
                 [[0], [1], [1], [1], [0]],
                 [[0], [1], [1], [1], [0]],
                 [[0], [1], [1], [1], [0]],
                 [[0], [0], [0], [0], [0]]]
        expected_output = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 2, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0
        ] * batch_size)
        # yapf: enable
        images = tf.constant([image] * batch_size, dtype=tf.uint8)
        for output_dtype in [tf.float16, tf.float32, tf.float64]:
            output = distance_tranform_ops.euclidean_dist_transform(
                images, dtype=output_dtype)
            output_flat = tf.reshape(output, [-1])

            with self.subTest(output_dtype=output_dtype):
                self.assertEqual(output.dtype, output_dtype)
                self.assertEqual(output.shape, [batch_size, 5, 5, 1])
                self.assertAllCloseAccordingToType(output_flat,
                                                   expected_output)

    def test_image_with_invalid_dtype(self):
        # yapf: disable
        image = [[[1], [1], [1], [1], [1]],
                 [[1], [1], [1], [1], [1]],
                 [[0], [1], [0], [1], [0]],
                 [[1], [0], [1], [0], [1]],
                 [[0], [1], [0], [1], [0]]]
        # yapf: enable
        image = tf.constant(image, dtype=tf.uint8)

        for output_dtype in [tf.uint8, tf.int32, tf.int64]:
            # pylint: disable=bad-continuation
            with self.assertRaisesRegex(
                    TypeError, "`dtype` must be float16, float32 or float64"):
                _ = distance_tranform_ops.euclidean_dist_transform(
                    image, dtype=output_dtype)

    def test_image_with_invalid_shape(self):
        for invalid_shape in ([1], [2, 1], [2, 4, 4, 4, 1]):
            image = tf.zeros(invalid_shape, tf.uint8)

            # pylint: disable=bad-continuation
            with self.assertRaisesRegex(
                    ValueError, "`images` should have rank between 3 and 4"):
                _ = distance_tranform_ops.euclidean_dist_transform(image)

        image = tf.zeros([2, 4, 3], tf.uint8)
        with self.assertRaisesRegex(ValueError,
                                    "`images` must have only one channel"):
            _ = distance_tranform_ops.euclidean_dist_transform(image)

    def test_all_zeros(self):
        image = tf.zeros([10, 10, 1], tf.uint8)
        expected_output = np.zeros([10, 10, 1])

        for output_dtype in [tf.float16, tf.float32, tf.float64]:
            output = distance_tranform_ops.euclidean_dist_transform(
                image, dtype=output_dtype)
            self.assertAllClose(output, expected_output)

    def test_all_ones(self):
        image = tf.ones([10, 10, 1], tf.uint8)
        output = distance_tranform_ops.euclidean_dist_transform(image)
        expected_output = np.full([10, 10, 1], tf.float32.max)
        self.assertAllClose(output, expected_output)


if __name__ == "__main__":
    tf.test.main()
