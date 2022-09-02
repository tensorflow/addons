# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from abc import abstractmethod
from typing import Optional

import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.python.autograph import set_loop_options
from typeguard import typechecked

from tensorflow_addons.utils.types import AcceptableDTypes


class StreamingBuffer(Metric):
    """`StreamingBuffer` is a base class to be used for metrics that have an
    algorithmic complexity that is too large to compute on the full dataset
    at once and that cannot be computed iteratively (e.g. mutual information).
    The class manages a buffer that produces chunks of data that are
    delivered to a child class implementing the metric computation logics.

    Child classes need to implement two abstract methods:

      - `_update_state(self, y_true_buffer, y_pred_buffer)` that updates the
      state of the metric given a batch of `y_true_buffer` and `y_pred_buffer`
      values with size `buffer_size`.

      - `_result(self)` that returns the metric's value.
    """

    @typechecked
    def __init__(
        self,
        buffer_size: int = 1024,
        name: Optional[str] = None,
        dtype: AcceptableDTypes = None,
    ):
        """Creates a `StreamingBuffer` instance."""
        super().__init__(name=name, dtype=dtype)

        self.max_buffer_size = buffer_size

        self._y_pred_buffer = self.add_weight(
            "y_pred_buffer", (self.max_buffer_size,), dtype=tf.float32
        )
        self._y_true_buffer = self.add_weight(
            "y_true_buffer", (self.max_buffer_size,), dtype=tf.float32
        )
        self._buffer_size = self.add_weight(
            "buffer_size", (), initializer="zeros", dtype=tf.int32
        )

    @property
    def y_pred_buffer(self):
        return self._y_pred_buffer[: self._buffer_size]

    @property
    def y_true_buffer(self):
        return self._y_true_buffer[: self._buffer_size]

    @abstractmethod
    def _update_state(self, y_true_buffer, y_pred_buffer):
        """Must be implemented by child class.
        Updates the state of the metric given a batch of `y_true_buffer`
        and `y_pred_buffer` values with size `self.buffer_size`."""
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Fills the buffer with flattened values of `y_true` and `y_pred`,
        and executes `_update_state(y_true_buffer, y_pred_buffer)` everytime
        it reaches maximum capacity.
        """
        flat_y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        flat_y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        def insert_data_in_buffer(y_true_data, y_pred_data):
            write_size = tf.minimum(
                self.max_buffer_size - self._buffer_size, tf.size(y_true_data)
            )
            indices = tf.range(write_size) + self._buffer_size
            labels_buffer = tf.tensor_scatter_nd_update(
                self._y_true_buffer,
                tf.expand_dims(indices, axis=-1),
                y_true_data[:write_size],
            )
            preds_buffer = tf.tensor_scatter_nd_update(
                self._y_pred_buffer,
                tf.expand_dims(indices, axis=-1),
                y_pred_data[:write_size],
            )
            self._y_true_buffer.assign(labels_buffer)
            self._y_pred_buffer.assign(preds_buffer)
            self._buffer_size.assign_add(write_size)
            return y_true_data[write_size:], y_pred_data[write_size:]

        labels_remainder, preds_remainder = insert_data_in_buffer(
            flat_y_true, flat_y_pred
        )

        while tf.size(labels_remainder) > 0:
            set_loop_options(
                shape_invariants=[
                    (labels_remainder, tf.TensorShape([None])),
                    (preds_remainder, tf.TensorShape([None])),
                ]
            )
            self._update_state(self.y_true_buffer, self.y_pred_buffer)
            self._buffer_size.assign(0)
            labels_remainder, preds_remainder = insert_data_in_buffer(
                labels_remainder, preds_remainder
            )

    @abstractmethod
    def _result(self):
        """Must be implemented by child class.
        Returns the metric's value."""
        pass

    def result(self):
        if self._buffer_size > 0:
            self._update_state(self.y_true_buffer, self.y_pred_buffer)
            self._buffer_size.assign(0)
        return self._result()

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {"buffer_size": self.max_buffer_size}
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        """Resets all of the metric state variables."""
        self._buffer_size.assign(0)

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()
