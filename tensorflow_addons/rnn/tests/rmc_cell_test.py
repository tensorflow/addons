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
"""Tests for RMC Cell."""

import pytest
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.rnn import RMCCell


@pytest.mark.skipif(
    tf.__version__[:3] == "2.3",
    reason="RMC will not work on Windows- and Linux-based distributions as it requires tf.keras.layers.MultiHeadAttention.",
)
def test_base_rmc():
    batch_size = 3
    units = 16

    expected_output = np.array(
        [
            [
                0.1684311,
                0.10384876,
                -0.03758289,
                -0.07105169,
                -0.28325063,
                0.34296486,
                -0.18999499,
                0.11927383,
                0.1684311,
                0.10384876,
                -0.03758289,
                -0.07105169,
                -0.28325063,
                0.34296486,
                -0.18999499,
                0.11927383,
            ],
            [
                0.1840748,
                0.09620976,
                -0.02045382,
                -0.07496377,
                -0.30143705,
                0.3523398,
                -0.21200192,
                0.13750899,
                0.1840748,
                0.09620976,
                -0.02045382,
                -0.07496377,
                -0.30143705,
                0.3523398,
                -0.21200192,
                0.13750899,
            ],
            [
                0.19798864,
                0.09918775,
                -0.01345633,
                -0.08062034,
                -0.31641114,
                0.3658627,
                -0.22976933,
                0.14713316,
                0.19798864,
                0.09918775,
                -0.01345633,
                -0.08062034,
                -0.31641114,
                0.3658627,
                -0.22976933,
                0.14713316,
            ],
        ],
        dtype=np.float32,
    )
    expected_state = np.array(
        [
            [
                0.1684311,
                0.10384876,
                -0.03758289,
                -0.07105169,
                -0.28325063,
                0.34296486,
                -0.18999499,
                0.11927383,
                0.1684311,
                0.10384876,
                -0.03758289,
                -0.07105169,
                -0.28325063,
                0.34296486,
                -0.18999499,
                0.11927383,
            ],
            [
                0.1840748,
                0.09620976,
                -0.02045382,
                -0.07496377,
                -0.30143705,
                0.3523398,
                -0.21200192,
                0.13750899,
                0.1840748,
                0.09620976,
                -0.02045382,
                -0.07496377,
                -0.30143705,
                0.3523398,
                -0.21200192,
                0.13750899,
            ],
            [
                0.19798864,
                0.09918775,
                -0.01345633,
                -0.08062034,
                -0.31641114,
                0.3658627,
                -0.22976933,
                0.14713316,
                0.19798864,
                0.09918775,
                -0.01345633,
                -0.08062034,
                -0.31641114,
                0.3658627,
                -0.22976933,
                0.14713316,
            ],
        ],
        dtype=np.float32,
    )

    const_initializer = tf.constant_initializer(0.1)
    cell = RMCCell(
        n_slots=2,
        n_heads=4,
        head_size=2,
        n_blocks=1,
        n_layers=3,
        activation=None,
        recurrent_activation=None,
        forget_bias=0.0,
        kernel_initializer=const_initializer,
        recurrent_initializer=const_initializer,
        bias_initializer=const_initializer,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
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

    # Asserts whether the outputs are being correctly calculated
    assert output.shape == (batch_size, units)
    np.testing.assert_allclose(output, expected_output, rtol=1e-6, atol=1e-6)

    # Asserts whether the states are being correctly calculated
    assert len(state) == 2
    h, m = state
    assert h.shape == (batch_size, units)
    assert m.shape == (batch_size, units)
    np.testing.assert_allclose(h, expected_state, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    tf.__version__[:3] == "2.3",
    reason="RMC will not work on Windows- and Linux-based distributions as it requires tf.keras.layers.MultiHeadAttention.",
)
def test_attend_over_memory_rmc():
    batch_size = 3
    n_slots = 2
    slot_size = 8

    cell = RMCCell(n_slots, 4, 2)

    seq_input = tf.convert_to_tensor(
        np.random.rand(batch_size, n_slots, slot_size), dtype=tf.float32
    )
    memory_value = tf.constant(
        np.ones((batch_size, n_slots, slot_size), dtype=np.float32), dtype=tf.float32
    )
    expected_att_memory = np.array(
        [
            [
                [
                    -0.6603111,
                    -0.4110478,
                    0.5193146,
                    1.2311147,
                    -1.0648061,
                    0.5235050,
                    1.3777641,
                    -1.5155334,
                ],
                [
                    -0.6603111,
                    -0.4110478,
                    0.5193146,
                    1.2311147,
                    -1.0648061,
                    0.52350503,
                    1.3777641,
                    -1.5155334,
                ],
            ],
            [
                [
                    -0.67101234,
                    -0.5338758,
                    0.65634435,
                    1.0622097,
                    -1.3630745,
                    0.6724906,
                    1.3870882,
                    -1.2101701,
                ],
                [
                    -0.67101234,
                    -0.5338758,
                    0.65634435,
                    1.0622097,
                    -1.3630745,
                    0.6724906,
                    1.3870882,
                    -1.2101701,
                ],
            ],
            [
                [
                    -0.72084284,
                    -0.47108832,
                    0.6819248,
                    1.0832782,
                    -1.162886,
                    0.5546384,
                    1.4232899,
                    -1.3883144,
                ],
                [
                    -0.72084284,
                    -0.47108832,
                    0.6819248,
                    1.0832782,
                    -1.162886,
                    0.5546384,
                    1.4232899,
                    -1.3883144,
                ],
            ],
        ],
        dtype=np.float32,
    )

    att_memory = cell._attend_over_memory(seq_input, memory_value)

    assert att_memory.shape == (batch_size, n_slots, slot_size)
    np.testing.assert_allclose(att_memory, expected_att_memory, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    tf.__version__[:3] == "2.3",
    reason="RMC will not work on Windows- and Linux-based distributions as it requires tf.keras.layers.MultiHeadAttention.",
)
def test_keras_rnn():
    batch_size = 2
    cell = RMCCell(2, 4, 2)

    seq_input = tf.convert_to_tensor(
        np.random.rand(batch_size, 3, 5), name="seq_input", dtype=tf.float32
    )

    rnn_layer = keras.layers.RNN(cell=cell)
    rnn_outputs = rnn_layer(
        seq_input,
        initial_state=cell.get_initial_state(seq_input, batch_size, dtype=tf.float32),
    )

    assert rnn_outputs.shape == (2, 16)


@pytest.mark.skipif(
    tf.__version__[:3] == "2.3",
    reason="RMC will not work on Windows- and Linux-based distributions as it requires tf.keras.layers.MultiHeadAttention.",
)
def test_config_rmc():
    cell = RMCCell(
        n_slots=2,
        n_heads=4,
        head_size=2,
        n_blocks=1,
        n_layers=3,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        forget_bias=1.0,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        name="rmc_cell_3",
    )

    expected_config = {
        "name": "rmc_cell_3",
        "trainable": True,
        "dtype": "float32",
        "n_slots": 2,
        "n_heads": 4,
        "head_size": 2,
        "n_blocks": 1,
        "n_layers": 3,
        "activation": tf.keras.activations.serialize(tf.keras.activations.get("tanh")),
        "recurrent_activation": tf.keras.activations.serialize(
            tf.keras.activations.get("hard_sigmoid")
        ),
        "forget_bias": 1.0,
        "kernel_initializer": tf.keras.initializers.serialize(
            tf.keras.initializers.get("glorot_uniform")
        ),
        "recurrent_initializer": tf.keras.initializers.serialize(
            tf.keras.initializers.get("orthogonal")
        ),
        "bias_initializer": tf.keras.initializers.serialize(
            tf.keras.initializers.get("zeros")
        ),
        "kernel_regularizer": tf.keras.regularizers.serialize(
            tf.keras.regularizers.get(None)
        ),
        "recurrent_regularizer": tf.keras.regularizers.serialize(
            tf.keras.regularizers.get(None)
        ),
        "bias_regularizer": tf.keras.regularizers.serialize(
            tf.keras.regularizers.get(None)
        ),
        "kernel_constraint": tf.keras.constraints.serialize(
            tf.keras.constraints.get(None)
        ),
        "recurrent_constraint": tf.keras.constraints.serialize(
            tf.keras.constraints.get(None)
        ),
        "bias_constraint": tf.keras.constraints.serialize(
            tf.keras.constraints.get(None)
        ),
    }
    config = cell.get_config()
    assert config == expected_config

    restored_cell = RMCCell.from_config(config)
    restored_config = restored_cell.get_config()
    assert config == restored_config
