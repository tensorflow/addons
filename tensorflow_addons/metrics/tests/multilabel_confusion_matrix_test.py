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
"""Tests for Multilabel Confusion Matrix Metric."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow_addons.metrics import MultiLabelConfusionMatrix


def test_config():
    mcm_obj = MultiLabelConfusionMatrix(num_classes=3)
    assert mcm_obj.num_classes == 3
    assert mcm_obj.dtype == tf.float32
    # Check save and restore config
    mcm_obj2 = MultiLabelConfusionMatrix.from_config(mcm_obj.get_config())
    assert mcm_obj2.num_classes == 3
    assert mcm_obj2.dtype == tf.float32


def check_results(obj, value):
    np.testing.assert_allclose(value, obj.result().numpy(), atol=1e-6)


@pytest.mark.parametrize("dtype", [tf.int32, tf.int64, tf.float32, tf.float64])
def test_mcm_3_classes(dtype):
    actuals = tf.constant([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype)
    preds = tf.constant([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1]], dtype=dtype)
    # Initialize
    mcm_obj = MultiLabelConfusionMatrix(num_classes=3, dtype=dtype)
    mcm_obj.update_state(actuals, preds)
    # Check results
    check_results(mcm_obj, [[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[0, 2], [2, 0]]])


@pytest.mark.parametrize("dtype", [tf.int32, tf.int64, tf.float32, tf.float64])
def test_mcm_4_classes(dtype):
    actuals = tf.constant(
        [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=dtype,
    )
    preds = tf.constant(
        [
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )

    # Initialize
    mcm_obj = MultiLabelConfusionMatrix(num_classes=4, dtype=dtype)
    mcm_obj.update_state(actuals, preds)
    # Check results
    check_results(
        mcm_obj,
        [[[4, 1], [1, 4]], [[6, 0], [2, 2]], [[6, 1], [1, 2]], [[2, 0], [2, 6]]],
    )


@pytest.mark.parametrize("dtype", [tf.int32, tf.int64, tf.float32, tf.float64])
def test_multiclass(dtype):
    actuals = tf.constant(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )
    preds = tf.constant(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )

    # Initialize
    mcm_obj = MultiLabelConfusionMatrix(num_classes=4, dtype=dtype)
    mcm_obj.update_state(actuals, preds)
    # Check results
    check_results(
        mcm_obj,
        [[[5, 2], [0, 3]], [[7, 1], [2, 0]], [[7, 0], [1, 2]], [[8, 0], [0, 2]]],
    )
