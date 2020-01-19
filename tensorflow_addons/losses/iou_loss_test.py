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
from tensorflow_addons.losses import iou_loss, IoULoss
from tensorflow_addons.losses import ciou_loss, CIoULoss
from tensorflow_addons.losses import diou_loss, DIoULoss
from tensorflow_addons.losses import giou_loss, GIoULoss


@test_utils.run_all_in_graph_and_eager_modes
class IoULossTest(tf.test.TestCase, parameterized.TestCase):
    """IoU losses test class."""

    def test_config(self):
        for Loss in [IoULoss, CIoULoss, DIoULoss, GIoULoss]:
            with self.subTest():
                loss = Loss(
                    reduction=tf.keras.losses.Reduction.NONE, name=Loss.__name__
                )
                self.assertEqual(loss.name, Loss.__name__)
                self.assertEqual(loss.reduction, tf.keras.losses.Reduction.NONE)

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_ious_loss(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype
        )
        losses = [iou_loss, ciou_loss, diou_loss, giou_loss]
        expected_results = [
            tf.constant(expected_result, dtype=dtype)
            for expected_result in [
                [0.875, 1.0],
                [1.4088933645154844, 1.5487535732151345],
                [1.4065315315315314, 1.5315315315315314],
                [1.07500000298023224, 1.9333333373069763],
            ]
        ]
        for iou_loss_imp, expected_result in zip(losses, expected_results):
            with self.subTest():
                loss = iou_loss_imp(boxes1, boxes2)
                self.assertAllCloseAccordingToType(loss, expected_result)

    def test_with_integer(self):
        boxes1 = tf.constant([[4, 3, 7, 5], [5, 6, 10, 7]], dtype=tf.int32)
        boxes2 = tf.constant([[3, 4, 6, 8], [14, 14, 15, 15]], dtype=tf.int32)
        expected_result = tf.constant(
            [1.07500000298023224, 1.9333333373069763], dtype=tf.float32
        )
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_different_shapes(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0]], dtype=dtype)
        expand_boxes1 = tf.expand_dims(boxes1, -2)
        expand_boxes2 = tf.expand_dims(boxes2, 0)
        losses = [iou_loss, ciou_loss, diou_loss, giou_loss]
        expected_results = [
            tf.constant(expected_result, dtype=dtype)
            for expected_result in [
                [0.875, 0.9375],
                [1.0117957952481038, 1.1123530805529542],
                [1.0094339622641511, 1.0719339622641511],
                [1.075, 1.3660714285714286],
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
        losses = [iou_loss, ciou_loss, diou_loss, giou_loss]
        expected_results = [
            tf.constant(expected_result, dtype=dtype)
            for expected_result in [
                0.875,
                0.999313052496148,
                0.99695121951219512,
                1.075,
            ]
        ]
        for iou_loss_imp, expected_result in zip(losses, expected_results):
            with self.subTest():
                loss = iou_loss_imp(boxes1, boxes2)
                self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_keras_model(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype
        )
        losses = [IoULoss, CIoULoss, DIoULoss, GIoULoss]
        expected_results = [
            tf.constant(expected_result, dtype=dtype)
            for expected_result in [
                [0.875, 1.0],
                [1.4088933645154844, 1.5487535732151345],
                [1.4065315315315314, 1.5315315315315314],
                [1.07500000298023224, 1.9333333373069763],
            ]
        ]
        for Loss, expected_result in zip(losses, expected_results):
            with self.subTest():
                model = tf.keras.Sequential()
                model.compile(
                    optimizer="adam",
                    loss=Loss(reduction=tf.keras.losses.Reduction.NONE),
                )
                loss = model.evaluate(boxes1, boxes2, batch_size=2, steps=1)
                self.assertAllCloseAccordingToType(loss, expected_result)


if __name__ == "__main__":
    tf.test.main()
