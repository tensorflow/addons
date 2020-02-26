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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils import test_utils
from tensorflow_addons.losses import giou_loss, GIoULoss


@test_utils.run_all_in_graph_and_eager_modes
class GIoULossTest(tf.test.TestCase, parameterized.TestCase):
    """GIoU test class."""

    def test_config(self):
        gl_obj = GIoULoss(reduction=tf.keras.losses.Reduction.NONE, name="giou_loss")
        self.assertEqual(gl_obj.name, "giou_loss")
        self.assertEqual(gl_obj.reduction, tf.keras.losses.Reduction.NONE)

    @parameterized.named_parameters(
        ("float16", np.float16), ("float32", np.float32), ("float64", np.float64)
    )
    def test_iou(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype
        )
        expected_result = tf.constant([0.875, 1.0], dtype=dtype)
        loss = giou_loss(boxes1, boxes2, mode="iou")
        self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(
        ("float16", np.float16), ("float32", np.float32), ("float64", np.float64)
    )
    def test_giou_loss(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype
        )
        expected_result = tf.constant(
            [1.07500000298023224, 1.9333333373069763], dtype=dtype
        )
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)

    def test_with_integer(self):
        boxes1 = tf.constant([[4, 3, 7, 5], [5, 6, 10, 7]], dtype=tf.int32)
        boxes2 = tf.constant([[3, 4, 6, 8], [14, 14, 15, 15]], dtype=tf.int32)
        expected_result = tf.constant(
            [1.07500000298023224, 1.9333333373069763], dtype=tf.float32
        )
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(
        ("float16", np.float16), ("float32", np.float32), ("float64", np.float64)
    )
    def test_different_shapes(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0]], dtype=dtype)
        tf.expand_dims(boxes1, -2)
        tf.expand_dims(boxes2, 0)
        expected_result = tf.constant([1.07500000298023224, 1.366071], dtype=dtype)
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)

        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0]], dtype=dtype)
        tf.expand_dims(boxes1, 0)
        tf.expand_dims(boxes2, -2)
        expected_result = tf.constant([1.07500000298023224, 1.366071], dtype=dtype)
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(
        ("float16", np.float16), ("float32", np.float32), ("float64", np.float64)
    )
    def test_one_bbox(self, dtype):
        boxes1 = tf.constant([4.0, 3.0, 7.0, 5.0], dtype=dtype)
        boxes2 = tf.constant([3.0, 4.0, 6.0, 8.0], dtype=dtype)
        expected_result = tf.constant(1.07500000298023224, dtype=dtype)
        loss = giou_loss(boxes1, boxes2)
        self.assertAllCloseAccordingToType(loss, expected_result)

    @parameterized.named_parameters(
        ("float16", np.float16), ("float32", np.float32), ("float64", np.float64)
    )
    def test_keras_model(self, dtype):
        boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
        boxes2 = tf.constant(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype
        )
        expected_result = tf.constant(
            [1.07500000298023224, 1.9333333373069763], dtype=dtype
        )
        model = tf.keras.Sequential()
        model.compile(
            optimizer="adam", loss=GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        )
        loss = model.evaluate(boxes1, boxes2, batch_size=2, steps=1)
        self.assertAllCloseAccordingToType(loss, expected_result)


if __name__ == "__main__":
    tf.test.main()
