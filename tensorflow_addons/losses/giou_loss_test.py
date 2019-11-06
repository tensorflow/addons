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
"""Tests for GIoU loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils import test_utils
from tensorflow_addons.losses import giou_loss, GIoULoss


@test_utils.run_all_in_graph_and_eager_modes
class GIoULossTest(tf.test.TestCase, parameterized.TestCase):
    """GIoU test class."""

    def test_config(self):
        gl_obj = GIoULoss(
            reduction=tf.keras.losses.Reduction.NONE, name='giou_loss')
        self.assertEqual(gl_obj.name, 'giou_loss')
        self.assertEqual(gl_obj.reduction, tf.keras.losses.Reduction.NONE)

    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_iou(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                             dtype=dtype)
        boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]],
                             dtype=dtype)
        expected_result = tf.constant([14.0 / 16.0, 1.], dtype=dtype)
        loss = giou_loss(boxes1, boxes2, mode='iou')
        self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_giou_loss(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                             dtype=dtype)
        boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]],
                             dtype=dtype)
        expected_result = tf.constant(
            [1.07500000298023224, 1.9333333373069763], dtype=dtype)
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_giou_loss_swapped(self, dtype):
        boxes1 = tf.constant([[7.0, 3.0, 4.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                             dtype=dtype)
        boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]],
                             dtype=dtype)
        expected_result = tf.constant(
            [1.07500000298023224, 1.9333333373069763], dtype=dtype)
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_giou_loss_with_dims(self, dtype):
        boxes1 = tf.constant([[[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                              [[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]]],
                             dtype=dtype)
        boxes2 = tf.constant(
            [[[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]],
             [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]]],
            dtype=dtype)
        expected_result = tf.constant(
            [[1.07500000298023224, 1.9333333373069763],
             [1.07500000298023224, 1.9333333373069763]],
            dtype=dtype)
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)


if __name__ == '__main__':
    tf.test.main()
