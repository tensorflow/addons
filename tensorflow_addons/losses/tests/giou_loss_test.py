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


import pytest

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils import test_utils
from tensorflow_addons.losses import giou_loss, GIoULoss


def test_config():
    gl_obj = GIoULoss(reduction=tf.keras.losses.Reduction.NONE, name="giou_loss")
    assert gl_obj.name == "giou_loss"
    assert gl_obj.reduction == tf.keras.losses.Reduction.NONE


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_iou(dtype):
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype)
    expected_result = tf.constant([0.875, 1.0], dtype=dtype)
    loss = giou_loss(boxes1, boxes2, mode="iou")
    test_utils.assert_allclose_according_to_type(loss, expected_result)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_giou_loss(dtype):
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype)
    expected_result = tf.constant(
        [1.07500000298023224, 1.9333333373069763], dtype=dtype
    )
    loss = giou_loss(boxes1, boxes2)
    test_utils.assert_allclose_according_to_type(loss, expected_result)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_different_shapes(dtype):
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0]], dtype=dtype)
    expand_boxes1 = tf.expand_dims(boxes1, -2)
    expand_boxes2 = tf.expand_dims(boxes2, 0)
    expected_result = tf.constant([1.07500000298023224, 1.366071], dtype=dtype)
    loss = giou_loss(expand_boxes1, expand_boxes2)
    test_utils.assert_allclose_according_to_type(loss, expected_result)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_one_bbox(dtype):
    boxes1 = tf.constant([4.0, 3.0, 7.0, 5.0], dtype=dtype)
    boxes2 = tf.constant([3.0, 4.0, 6.0, 8.0], dtype=dtype)
    expected_result = tf.constant(1.07500000298023224, dtype=dtype)
    loss = giou_loss(boxes1, boxes2)
    test_utils.assert_allclose_according_to_type(loss, expected_result)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_keras_model(dtype):
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=dtype)
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=dtype)
    expected_result = tf.constant(1.5041667222976685, dtype=dtype)
    model = tf.keras.Sequential()
    model.compile(
        optimizer="adam",
        loss=GIoULoss(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
    )
    loss = model.evaluate(boxes1, boxes2, batch_size=2, steps=1)
    test_utils.assert_allclose_according_to_type(loss, expected_result)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_with_integer():
    boxes1 = tf.constant([[4, 3, 7, 5], [5, 6, 10, 7]], dtype=tf.int32)
    boxes2 = tf.constant([[3, 4, 6, 8], [14, 14, 15, 15]], dtype=tf.int32)
    expected_result = tf.constant(
        [1.07500000298023224, 1.9333333373069763], dtype=tf.float32
    )
    loss = giou_loss(boxes1, boxes2)
    test_utils.assert_allclose_according_to_type(loss, expected_result)
