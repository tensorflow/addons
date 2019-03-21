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

# yapf: disable
_IMAGE = [[1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [0, 1, 0, 1, 0],
          [1, 0, 1, 0, 1],
          [0, 1, 0, 1, 0]]
# yapf: enable
_GROUND_TRUTH = [
    2, 2.23606801, 2, 2.23606801, 2, 1, 1.41421354, 1, 1.41421354, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
]


@test_utils.run_all_in_graph_and_eager_modes
class DistanceOpsTest(tf.test.TestCase):
    def test_single_img(self):
        for dtype in [tf.float16, tf.float32, tf.float64]:
            image = tf.constant(_IMAGE, dtype=tf.uint8)
            image = tf.reshape(image, [5, 5, 1])

            output = distance_tranform_ops.euclidean_dist_transform(
                image, dtype=dtype)
            output_flat = tf.reshape(output, [-1])
            expected_output = np.array(_GROUND_TRUTH)

            with self.subTest(name="output_value"):
                self.assertAllCloseAccordingToType(output_flat,
                                                   expected_output)
            with self.subTest(name="output_type"):
                self.assertEqual(output.dtype, dtype)
            with self.subTest(name="output_shape"):
                self.assertEqual(output.shape, [5, 5, 1])

    def test_multiple_imgs(self):
        for dtype in [tf.float16, tf.float32, tf.float64]:
            image = tf.constant(_IMAGE, dtype=tf.uint8)
            image = tf.reshape(image, [5, 5, 1])
            images = tf.stack([image, image, image], axis=0)

            output = distance_tranform_ops.euclidean_dist_transform(
                images, dtype=dtype)
            output_flat = tf.reshape(output, [-1])
            expected_output = np.array(_GROUND_TRUTH * 3)

            with self.subTest(name="output_value"):
                self.assertAllCloseAccordingToType(output_flat,
                                                   expected_output)
            with self.subTest(name="output_type"):
                self.assertEqual(output.dtype, dtype)
            with self.subTest(name="output_shape"):
                self.assertEqual(output.shape, [3, 5, 5, 1])

    def test_failure_dtype(self):
        for dtype in [tf.uint8, tf.int32, tf.int64]:
            image = tf.constant(_IMAGE, dtype=tf.uint8)
            image = tf.reshape(image, [5, 5, 1])

            with self.assertRaises(TypeError):
                output = distance_tranform_ops.euclidean_dist_transform(
                    image, dtype=dtype)

    def test_failure_shape(self):
        for shape in ([10, 10], [1, 2, 3, 4, 5], [100, 100, 3]):
            image = tf.zeros(shape, tf.uint8)
            with self.assertRaises(ValueError):
                output = distance_tranform_ops.euclidean_dist_transform(image)

    def test_all_zeroes(self):
        image = tf.zeros([10, 10, 1], tf.uint8)
        for dtype in [tf.float16, tf.float32, tf.float64]:
            output = distance_tranform_ops.euclidean_dist_transform(
                image, dtype)
            expected_output = np.zeros([10, 10, 1])

        self.assertAllClose(output, expected_output)

    def test_all_ones(self):
        image = tf.ones([10, 10, 1], tf.uint8)
        output = distance_tranform_ops.euclidean_dist_transform(image)
        expected_output = np.full([10, 10, 1], tf.float32.max)
        self.assertAllClose(output, expected_output)


if __name__ == "__main__":
    tf.test.main()
