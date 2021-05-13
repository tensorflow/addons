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
"""Tests for NAS Cell."""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.rnn import NASCell


def test_base():
    units = 6
    batch_size = 3
    expected_output = np.array(
        [
            [0.576751, 0.576751, 0.576751, 0.576751, 0.576751, 0.576751],
            [0.618936, 0.618936, 0.618936, 0.618936, 0.618936, 0.618936],
            [0.627393, 0.627393, 0.627393, 0.627393, 0.627393, 0.627393],
        ]
    )
    expected_state = np.array(
        [
            [
                0.7157977,
                0.7157977,
                0.7157977,
                0.7157977,
                0.7157977,
                0.7157977,
                0.5767508,
                0.5767508,
                0.5767508,
                0.5767508,
                0.5767508,
                0.5767508,
            ],
            [
                0.7804162,
                0.7804162,
                0.7804162,
                0.7804162,
                0.7804162,
                0.7804162,
                0.6189357,
                0.6189357,
                0.6189357,
                0.6189357,
                0.6189357,
                0.6189357,
            ],
            [
                0.7945764,
                0.7945764,
                0.7945764,
                0.7945764,
                0.7945765,
                0.7945765,
                0.6273934,
                0.6273934,
                0.6273934,
                0.6273934,
                0.6273934,
                0.6273934,
            ],
        ]
    )
    const_initializer = tf.constant_initializer(0.5)
    cell = NASCell(
        units=units,
        kernel_initializer=const_initializer,
        recurrent_initializer=const_initializer,
    )

    inputs = tf.constant(
        np.array(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            dtype=np.float32,
        ),
        dtype=tf.float32,
    )
    state_value = tf.constant(
        0.1 * np.ones((batch_size, units), dtype=np.float32), dtype=tf.float32
    )
    init_state = [state_value, state_value]
    output, state = cell(inputs, init_state)
    res = [output, state]

    # This is a smoke test: Only making sure expected values not change.
    assert len(res) == 2
    np.testing.assert_allclose(res[0], expected_output, rtol=1e-6, atol=1e-6)
    # There should be 2 states in the list.
    assert len(res[1]) == 2
    # Checking the shape of each state to be batch_size * num_units
    new_c, new_h = res[1]
    assert new_c.shape[0] == batch_size
    assert new_c.shape[1] == units
    assert new_h.shape[0] == batch_size
    assert new_h.shape[1] == units
    np.testing.assert_allclose(
        np.concatenate(res[1], axis=1), expected_state, rtol=1e-6, atol=1e-6
    )


def test_projection():
    units = 6
    batch_size = 3
    projection = 5
    expected_output = np.array(
        [
            [1.697418, 1.697418, 1.697418, 1.697418, 1.697418],
            [1.840037, 1.840037, 1.840037, 1.840037, 1.840037],
            [1.873985, 1.873985, 1.873985, 1.873985, 1.873985],
        ]
    )

    expected_state = np.array(
        [
            [
                0.69855207,
                0.69855207,
                0.69855207,
                0.69855207,
                0.69855207,
                0.69855207,
                1.69741797,
                1.69741797,
                1.69741797,
                1.69741797,
                1.69741797,
            ],
            [
                0.77073824,
                0.77073824,
                0.77073824,
                0.77073824,
                0.77073824,
                0.77073824,
                1.84003687,
                1.84003687,
                1.84003687,
                1.84003687,
                1.84003687,
            ],
            [
                0.78973997,
                0.78973997,
                0.78973997,
                0.78973997,
                0.78973997,
                0.78973997,
                1.87398517,
                1.87398517,
                1.87398517,
                1.87398517,
                1.87398517,
            ],
        ]
    )
    const_initializer = tf.constant_initializer(0.5)
    cell = NASCell(
        units=units,
        projection=projection,
        kernel_initializer=const_initializer,
        recurrent_initializer=const_initializer,
        projection_initializer=const_initializer,
    )
    inputs = tf.constant(
        np.array(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            dtype=np.float32,
        ),
        dtype=tf.float32,
    )
    state_value_c = tf.constant(
        0.1 * np.ones((batch_size, units), dtype=np.float32), dtype=tf.float32
    )
    state_value_h = tf.constant(
        0.1 * np.ones((batch_size, projection), dtype=np.float32), dtype=tf.float32
    )
    init_state = [state_value_c, state_value_h]
    output, state = cell(inputs, init_state)
    res = [output, state]

    # This is a smoke test: Only making sure expected values not change.
    assert len(res) == 2
    np.testing.assert_allclose(res[0], expected_output, rtol=1e-6, atol=1e-6)
    # There should be 2 states in the tuple.
    assert len(res[1]) == 2
    # Checking the shape of each state to be batch_size * num_units
    new_c, new_h = res[1]
    assert new_c.shape[0] == batch_size
    assert new_c.shape[1] == units
    assert new_h.shape[0] == batch_size
    assert new_h.shape[1] == projection
    np.testing.assert_allclose(
        np.concatenate(res[1], axis=1), expected_state, rtol=1e-6, atol=1e-6
    )


def test_keras_rnn():
    """Tests that NASCell works with keras RNN layer."""
    cell = NASCell(10)
    seq_input = tf.convert_to_tensor(
        np.random.rand(2, 3, 5), name="seq_input", dtype=tf.float32
    )
    rnn_layer = keras.layers.RNN(cell=cell)
    rnn_outputs = rnn_layer(seq_input)
    assert rnn_outputs.shape == (2, 10)


def test_config_nas():
    cell = NASCell(10, projection=5, use_bias=True, name="nas_cell_3")

    expected_config = {
        "dtype": "float32",
        "name": "nas_cell_3",
        "trainable": True,
        "units": 10,
        "projection": 5,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "recurrent_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "projection_initializer": "glorot_uniform",
    }
    config = cell.get_config()
    assert config == expected_config

    restored_cell = NASCell.from_config(config)
    restored_config = restored_cell.get_config()
    assert config == restored_config
