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
"""Tests for Peephole Cell."""

import numpy as np
import tensorflow as tf

from tensorflow_addons.rnn import PeepholeLSTMCell


def test_peephole_lstm_cell():
    def _run_cell(cell_fn, **kwargs):
        inputs = tf.one_hot([1, 2, 3, 4], 4)
        cell = cell_fn(5, **kwargs)
        cell.build(inputs.shape)
        initial_state = cell.get_initial_state(
            inputs=inputs, batch_size=4, dtype=tf.float32
        )
        output, _ = cell(inputs, initial_state)
        return output

    tf.random.set_seed(12345)
    first_implementation_output = _run_cell(
        PeepholeLSTMCell,
        kernel_initializer="ones",
        recurrent_activation="sigmoid",
        implementation=1,
    )
    second_implementation_output = _run_cell(
        PeepholeLSTMCell,
        kernel_initializer="ones",
        recurrent_activation="sigmoid",
        implementation=2,
    )
    expected_output = np.asarray(
        [
            [0.417551, 0.417551, 0.417551, 0.417551, 0.417551],
            [0.417551, 0.417551, 0.417551, 0.417551, 0.417551],
            [0.417551, 0.417551, 0.417551, 0.417551, 0.417551],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        first_implementation_output, second_implementation_output
    )
    np.testing.assert_allclose(
        first_implementation_output, expected_output, rtol=1e-6, atol=1e-6
    )
