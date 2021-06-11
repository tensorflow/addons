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
"""Tests for LayerNormLSTM Cell."""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.rnn import LayerNormLSTMCell


def test_cell_output():
    x = tf.ones([1, 2], dtype=tf.float32)
    c0 = tf.constant(0.1 * np.asarray([[0, 1]]), dtype=tf.float32)
    h0 = tf.constant(0.1 * np.asarray([[2, 3]]), dtype=tf.float32)
    state0 = [h0, c0]
    c1 = tf.constant(0.1 * np.asarray([[4, 5]]), dtype=tf.float32)
    h1 = tf.constant(0.1 * np.asarray([[6, 7]]), dtype=tf.float32)
    state1 = [h1, c1]
    state = (state0, state1)
    const_initializer = tf.constant_initializer(0.5)

    def single_cell():
        return LayerNormLSTMCell(
            units=2,
            kernel_initializer=const_initializer,
            recurrent_initializer=const_initializer,
            bias_initializer=const_initializer,
            norm_epsilon=1e-12,
        )

    cell = keras.layers.StackedRNNCells([single_cell() for _ in range(2)])
    output_v, output_states_v = cell(x, state)

    expected_output = np.array([[-0.47406167, 0.47406143]])
    expected_state0_c = np.array([[-1.0, 1.0]])
    expected_state0_h = np.array([[-0.47406167, 0.47406143]])
    expected_state1_c = np.array([[-1.0, 1.0]])
    expected_state1_h = np.array([[-0.47406167, 0.47406143]])

    actual_state0_h = output_states_v[0][0]
    actual_state0_c = output_states_v[0][1]
    actual_state1_h = output_states_v[1][0]
    actual_state1_c = output_states_v[1][1]

    np.testing.assert_allclose(output_v, expected_output, 1e-5)
    np.testing.assert_allclose(expected_state0_c, actual_state0_c, 1e-5)
    np.testing.assert_allclose(expected_state0_h, actual_state0_h, 1e-5)
    np.testing.assert_allclose(expected_state1_c, actual_state1_c, 1e-5)
    np.testing.assert_allclose(expected_state1_h, actual_state1_h, 1e-5)

    # Test BasicLSTMCell with input_size != num_units.
    x = tf.ones([1, 3], dtype=tf.float32)
    c = tf.constant(0.1 * np.asarray([[0, 1]]), dtype=tf.float32)
    h = tf.constant(0.1 * np.asarray([[2, 3]]), dtype=tf.float32)
    state = [h, c]
    cell = LayerNormLSTMCell(
        units=2,
        kernel_initializer=const_initializer,
        recurrent_initializer=const_initializer,
        bias_initializer=const_initializer,
        norm_epsilon=1e-12,
    )
    output_v, output_states_v = cell(x, state)
    expected_h = np.array([[-0.47406167, 0.47406143]])
    expected_c = np.array([[-1.0, 1.0]])
    np.testing.assert_allclose(output_v, expected_h, 1e-5)
    np.testing.assert_allclose(output_states_v[0], expected_h, 1e-5)
    np.testing.assert_allclose(output_states_v[1], expected_c, 1e-5)


def test_config_layer_norm():
    cell = LayerNormLSTMCell(10, name="layer_norm_lstm_cell_3")

    expected_config = {
        "dtype": "float32",
        "name": "layer_norm_lstm_cell_3",
        "trainable": True,
        "units": 10,
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "use_bias": True,
        "kernel_initializer": {
            "class_name": "GlorotUniform",
            "config": {"seed": None},
        },
        "recurrent_initializer": {
            "class_name": "Orthogonal",
            "config": {"seed": None, "gain": 1.0},
        },
        "bias_initializer": {"class_name": "Zeros", "config": {}},
        "unit_forget_bias": True,
        "kernel_regularizer": None,
        "recurrent_regularizer": None,
        "bias_regularizer": None,
        "kernel_constraint": None,
        "recurrent_constraint": None,
        "bias_constraint": None,
        "dropout": 0.0,
        "recurrent_dropout": 0.0,
        "implementation": 2,
        "norm_gamma_initializer": {"class_name": "Ones", "config": {}},
        "norm_beta_initializer": {"class_name": "Zeros", "config": {}},
        "norm_epsilon": 1e-3,
    }
    config = cell.get_config()
    assert config == expected_config

    restored_cell = LayerNormLSTMCell.from_config(config)
    restored_config = restored_cell.get_config()
    assert config == restored_config


def test_build():
    cell = LayerNormLSTMCell(10, name="layer_norm_lstm_cell")
    cell(
        inputs=tf.ones((12, 20)),
        states=cell.get_initial_state(batch_size=12, dtype=tf.float32),
    )
    assert len(cell.weights) == 9
    assert cell.weights[0].name == "layer_norm_lstm_cell/kernel:0"
    assert cell.weights[1].name == "layer_norm_lstm_cell/recurrent_kernel:0"
    assert cell.weights[2].name == "layer_norm_lstm_cell/bias:0"
    assert cell.weights[3].name == "layer_norm_lstm_cell/kernel_norm/gamma:0"
    assert cell.weights[4].name == "layer_norm_lstm_cell/kernel_norm/beta:0"
    assert cell.weights[5].name == "layer_norm_lstm_cell/recurrent_norm/gamma:0"
    assert cell.weights[6].name == "layer_norm_lstm_cell/recurrent_norm/beta:0"
    assert cell.weights[7].name == "layer_norm_lstm_cell/state_norm/gamma:0"
    assert cell.weights[8].name == "layer_norm_lstm_cell/state_norm/beta:0"
