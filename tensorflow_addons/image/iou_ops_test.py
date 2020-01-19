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
"""Tests for IoU losses."""

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils import test_utils
from tensorflow_addons.image import iou, ciou, diou, giou


@test_utils.run_all_in_graph_and_eager_modes
class IoUTest(tf.test.TestCase, parameterized.TestCase):
    """IoU test class."""

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_ious_loss(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype
        )
        losses = [iou, ciou, diou, giou]
        expected_results = [
            tf.constant(expected_result, dtype=dtype)
            for expected_result in [
                [0.125, 0.0],
                [-0.4088933645154844, -0.5487535732151345],
                [-0.4065315315315314, -0.5315315315315314],
                [-0.07500000298023224, -0.9333333373069763],
            ]
        ]
        for iou_loss_imp, expected_result in zip(losses, expected_results):
            with self.subTest():
                loss = iou_loss_imp(boxes1, boxes2)
                self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_different_shapes(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0]], dtype=dtype)
        expand_boxes1 = tf.expand_dims(boxes1, -2)
        expand_boxes2 = tf.expand_dims(boxes2, 0)
        losses = [iou, ciou, diou, giou]
        expected_results = [
            tf.constant(expected_result, dtype=dtype)
            for expected_result in [
                [0.125, 0.0625],
                [-0.0117957952481038, -0.1123530805529542],
                [-0.0094339622641511, -0.0719339622641511],
                [-0.075, -0.3660714285714286],
            ]
        ]
        for iou_loss_imp, expected_result in zip(losses, expected_results):
            with self.subTest():
                loss = iou_loss_imp(expand_boxes1, expand_boxes2)
                self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_one_bbox(self, dtype):
        boxes1 = tf.constant([4.0, 3.0, 7.0, 5.0], dtype=dtype)
        boxes2 = tf.constant([3.0, 4.0, 6.0, 8.0], dtype=dtype)
        losses = [iou, ciou, diou, giou]
        expected_results = [
            tf.constant(expected_result, dtype=dtype)
            for expected_result in [0.125, 0.000686947503852, 0.0030487804878, -0.075]
        ]
        for iou_loss_imp, expected_result in zip(losses, expected_results):
            with self.subTest():
                loss = iou_loss_imp(boxes1, boxes2)
                self.assertAllCloseAccordingToType(loss, expected_result)


if __name__ == "__main__":
    tf.test.main()
