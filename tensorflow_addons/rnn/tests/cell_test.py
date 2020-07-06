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
"""Tests for RNN cells."""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.rnn import cell as rnn_cell
from tensorflow_addons.rnn import LayerNormSimpleRNNCell


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
    cell = rnn_cell.NASCell(
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
    cell = rnn_cell.NASCell(
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
    cell = rnn_cell.NASCell(10)
    seq_input = tf.convert_to_tensor(
        np.random.rand(2, 3, 5), name="seq_input", dtype=tf.float32
    )
    rnn_layer = keras.layers.RNN(cell=cell)
    rnn_outputs = rnn_layer(seq_input)
    assert rnn_outputs.shape == (2, 10)


def test_config_nas():
    cell = rnn_cell.NASCell(10, projection=5, use_bias=True, name="nas_cell_3")

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

    restored_cell = rnn_cell.NASCell.from_config(config)
    restored_config = restored_cell.get_config()
    assert config == restored_config


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
        return rnn_cell.LayerNormLSTMCell(
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
    cell = rnn_cell.LayerNormLSTMCell(
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
    cell = rnn_cell.LayerNormLSTMCell(10, name="layer_norm_lstm_cell_3")

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

    restored_cell = rnn_cell.LayerNormLSTMCell.from_config(config)
    restored_config = restored_cell.get_config()
    assert config == restored_config


def test_constraints_layernorm_rnn():
    embedding_dim = 4
    k_constraint = keras.constraints.max_norm(0.01)
    r_constraint = keras.constraints.max_norm(0.01)
    b_constraint = keras.constraints.max_norm(0.01)
    g_constraint = keras.constraints.max_norm(0.01)
    layer = keras.layers.RNN(
        LayerNormSimpleRNNCell(
            units=5,
            kernel_constraint=k_constraint,
            recurrent_constraint=r_constraint,
            bias_constraint=b_constraint,
            gamma_constraint=g_constraint,
        ),
        input_shape=(None, embedding_dim),
        return_sequences=False,
    )
    layer.build((None, None, embedding_dim))
    assert layer.cell.kernel.constraint == k_constraint
    assert layer.cell.recurrent_kernel.constraint == r_constraint
    assert layer.cell.bias.constraint == b_constraint
    assert layer.cell.layernorm.gamma.constraint == g_constraint


def test_with_masking_layer_layernorm_rnn():
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(input_shape=(3, 4)))
    model.add(
        keras.layers.RNN(
            LayerNormSimpleRNNCell(units=5), return_sequences=True, unroll=False
        )
    )
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)


def test_regularizers_layernorm_rnn():
    embedding_dim = 4
    layer = keras.layers.RNN(
        LayerNormSimpleRNNCell(
            units=5,
            kernel_regularizer=keras.regularizers.l1(0.01),
            recurrent_regularizer=keras.regularizers.l1(0.01),
            bias_regularizer="l2",
            gamma_regularizer="l2",
        ),
        input_shape=(None, embedding_dim),
        return_sequences=False,
    )
    layer.build((None, None, 2))
    assert len(layer.losses) == 4


def test_configs_layernorm():
    config = {"layernorm_epsilon": 1e-6}
    cell1 = LayerNormSimpleRNNCell(units=8, **config)
    config1 = cell1.get_config()
    cell2 = LayerNormSimpleRNNCell(**config1)
    config2 = cell2.get_config()
    assert config1 == config2


def test_base_esn():
    units = 3
    expected_output = np.array(
        [[2.77, 2.77, 2.77], [4.77, 4.77, 4.77], [6.77, 6.77, 6.77],], dtype=np.float32,
    )

    const_initializer = tf.constant_initializer(0.5)
    cell = rnn_cell.ESNCell(
        units=units,
        connectivity=1,
        leaky=1,
        spectral_radius=0.9,
        use_norm2=True,
        use_bias=True,
        activation=None,
        kernel_initializer=const_initializer,
        recurrent_initializer=const_initializer,
        bias_initializer=const_initializer,
    )

    inputs = tf.constant(
        np.array(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            dtype=np.float32,
        ),
        dtype=tf.float32,
    )
    state_value = tf.constant(
        0.3 * np.ones((units, units), dtype=np.float32), dtype=tf.float32
    )
    init_state = [state_value, state_value]
    output, state = cell(inputs, init_state)

    np.testing.assert_allclose(output, expected_output, 1e-5)
    np.testing.assert_allclose(state, expected_output, 1e-5)


def test_esn_echo_state_property_eig():
    use_norm2 = False
    units = 3
    cell = rnn_cell.ESNCell(
        units=units,
        use_norm2=use_norm2,
        recurrent_initializer="ones",
        connectivity=1.0,
    )
    cell.build((3, 3))
    recurrent_weights = tf.constant(cell.get_weights()[0], dtype=tf.float32)
    max_eig = tf.reduce_max(tf.abs(tf.linalg.eig(recurrent_weights)[0]))
    assert max_eig < 1, "max(eig(W)) < 1"


def test_esn_echo_state_property_norm2():
    use_norm2 = True
    units = 3
    cell = rnn_cell.ESNCell(
        units=units, use_norm2=use_norm2, recurrent_initializer="ones", connectivity=1.0
    )
    cell.build((3, 3))
    recurrent_weights = tf.constant(cell.get_weights()[0])
    max_eig = tf.reduce_max(tf.abs(tf.linalg.eig(recurrent_weights)[0]))
    assert max_eig < 1, "max(eig(W)) < 1"


def test_esn_connectivity():
    units = 1000
    connectivity = 0.5
    cell = rnn_cell.ESNCell(
        units=units,
        connectivity=connectivity,
        use_norm2=True,
        recurrent_initializer="ones",
    )
    cell.build((3, 3))
    recurrent_weights = tf.constant(cell.get_weights()[0])
    num_non_zero = tf.math.count_nonzero(recurrent_weights)
    actual_connectivity = tf.divide(num_non_zero, units ** 2)
    np.testing.assert_allclose(
        np.asarray([actual_connectivity]), np.asanyarray([connectivity]), 1e-2
    )


def test_esn_keras_rnn():
    cell = rnn_cell.ESNCell(10)
    seq_input = tf.convert_to_tensor(
        np.random.rand(2, 3, 5), name="seq_input", dtype=tf.float32
    )
    rnn_layer = keras.layers.RNN(cell=cell)
    rnn_outputs = rnn_layer(seq_input)
    assert rnn_outputs.shape == (2, 10)


def test_esn_keras_rnn_e2e():
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    cell = rnn_cell.ESNCell(5)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(input_shape=(3, 4)))
    model.add(keras.layers.RNN(cell))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)


def test_esn_config():
    cell = rnn_cell.ESNCell(
        units=3,
        connectivity=1,
        leaky=1,
        spectral_radius=0.9,
        use_norm2=False,
        use_bias=True,
        activation="tanh",
        kernel_initializer="glorot_uniform",
        recurrent_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
        name="esn_cell_3",
    )

    expected_config = {
        "name": "esn_cell_3",
        "trainable": True,
        "dtype": "float32",
        "units": 3,
        "connectivity": 1,
        "leaky": 1,
        "spectral_radius": 0.9,
        "use_norm2": False,
        "use_bias": True,
        "activation": tf.keras.activations.serialize(tf.keras.activations.get("tanh")),
        "kernel_initializer": tf.keras.initializers.serialize(
            tf.keras.initializers.get("glorot_uniform")
        ),
        "recurrent_initializer": tf.keras.initializers.serialize(
            tf.keras.initializers.get("glorot_uniform")
        ),
        "bias_initializer": tf.keras.initializers.serialize(
            tf.keras.initializers.get("glorot_uniform")
        ),
    }
    config = cell.get_config()
    assert config == expected_config

    restored_cell = rnn_cell.ESNCell.from_config(config)
    restored_config = restored_cell.get_config()
    assert config == restored_config


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
        rnn_cell.PeepholeLSTMCell,
        kernel_initializer="ones",
        recurrent_activation="sigmoid",
        implementation=1,
    )
    second_implementation_output = _run_cell(
        rnn_cell.PeepholeLSTMCell,
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
