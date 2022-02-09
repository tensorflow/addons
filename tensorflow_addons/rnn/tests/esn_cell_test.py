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
"""Tests for ESN Cell."""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.rnn import ESNCell


def test_base_esn():
    units = 3
    expected_output = np.array(
        [[2.77, 2.77, 2.77], [4.77, 4.77, 4.77], [6.77, 6.77, 6.77]], dtype=np.float32
    )

    const_initializer = tf.constant_initializer(0.5)
    cell = ESNCell(
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
    cell = ESNCell(
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
    cell = ESNCell(
        units=units, use_norm2=use_norm2, recurrent_initializer="ones", connectivity=1.0
    )
    cell.build((3, 3))
    recurrent_weights = tf.constant(cell.get_weights()[0])
    max_eig = tf.reduce_max(tf.abs(tf.linalg.eig(recurrent_weights)[0]))
    assert max_eig < 1, "max(eig(W)) < 1"


def test_esn_connectivity():
    units = 1000
    connectivity = 0.5
    cell = ESNCell(
        units=units,
        connectivity=connectivity,
        use_norm2=True,
        recurrent_initializer="ones",
    )
    cell.build((3, 3))
    recurrent_weights = tf.constant(cell.get_weights()[0])
    num_non_zero = tf.math.count_nonzero(recurrent_weights)
    actual_connectivity = tf.divide(num_non_zero, units**2)
    np.testing.assert_allclose(
        np.asarray([actual_connectivity]), np.asanyarray([connectivity]), 1e-2
    )


def test_esn_keras_rnn():
    cell = ESNCell(10)
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
    cell = ESNCell(5)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(input_shape=(3, 4)))
    model.add(keras.layers.RNN(cell))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)


def test_esn_config():
    cell = ESNCell(
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

    restored_cell = ESNCell.from_config(config)
    restored_config = restored_cell.get_config()
    assert config == restored_config
