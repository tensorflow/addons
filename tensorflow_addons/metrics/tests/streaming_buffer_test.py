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
"""Tests for StreamingBuffer."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.metrics._streaming_buffer import StreamingBuffer
from tensorflow_addons.testing.serialization import check_metric_serialization


class Counter(StreamingBuffer):
    def __init__(self, buffer_size: int = 1024, name=None, dtype=None):
        super().__init__(buffer_size, name, dtype)
        self.pred_count = self.add_weight("pred_count", (), dtype=tf.int32)
        self.true_count = self.add_weight("true_count", (), dtype=tf.int32)
        self.update_count = self.add_weight("update_count", (), dtype=tf.int32)

    def _update_state(self, y_true_buffer, y_pred_buffer):
        self.true_count.assign_add(tf.reduce_sum(tf.cast(y_true_buffer, tf.int32)))
        self.pred_count.assign_add(tf.reduce_sum(tf.cast(y_pred_buffer, tf.int32)))
        self.update_count.assign_add(1)

    def _result(self):
        return tf.concat([self.true_count, self.pred_count, self.update_count], axis=0)

    def reset_state(self):
        self.true_count.assign(0)
        self.pred_count.assign(0)
        self.update_count.assign(0)


@pytest.mark.parametrize(
    "buffer_size, dataset_size, batch_size",
    [
        (8, 16, 4),
        (16, 16, 4),
        (16, 16, 16),
        (16, 15, 4),
        (15, 16, 4),
        (15, 16, 3),
    ],
)
def test_buffer(buffer_size, dataset_size, batch_size):
    metric = Counter(buffer_size=buffer_size)
    data = np.ones((dataset_size, 2))
    x, y = data[:, 0], data[:, 1]
    dataset = tf.data.Dataset.from_tensor_slices({"x": x, "y": y}).batch(
        batch_size, drop_remainder=False
    )
    for batch in dataset:
        metric.update_state(batch["x"], batch["y"])

    for _ in range(2):
        # multiple calls to `results` must be idempotent
        result = metric.result()
        assert result[0] == result[1]
        assert result[0] == dataset_size
        assert result[2] == np.ceil(dataset_size / buffer_size)


def test_serialization():
    labels = np.array([4, 4, 3, 3, 2, 2, 1, 1], dtype=np.int32)
    preds = np.array([1, 2, 4, 1, 3, 3, 4, 4], dtype=np.int32)

    kt = Counter(buffer_size=4)
    check_metric_serialization(kt, labels, preds)
