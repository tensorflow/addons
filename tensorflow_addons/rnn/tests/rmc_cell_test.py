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

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.rnn import RMCCell

def test_keras_rnn():
    """Tests that RMCCell works with keras RNN layer."""
    cell = RMCCell(2, 4, 2)
    seq_input = tf.convert_to_tensor(
        np.random.rand(2, 3, 5), name="seq_input", dtype=tf.float32
    )
    rnn_layer = keras.layers.RNN(cell=cell)
    rnn_outputs = rnn_layer(seq_input)
    assert rnn_outputs.shape == (2, 16)


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
        name="rmc_cell_3")

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
        "recurrent_activation": tf.keras.activations.serialize(tf.keras.activations.get("hard_sigmoid")),
        "forget_bias": 1.0,
        "kernel_initializer": tf.keras.initializers.serialize(tf.keras.initializers.get("glorot_uniform")),
        "recurrent_initializer": tf.keras.initializers.serialize(tf.keras.initializers.get("orthogonal")),
        "bias_initializer": tf.keras.initializers.serialize(tf.keras.initializers.get("zeros")),
        "kernel_regularizer": tf.keras.regularizers.serialize(tf.keras.regularizers.get(None)),
        "recurrent_regularizer": tf.keras.regularizers.serialize(tf.keras.regularizers.get(None)),
        "bias_regularizer": tf.keras.regularizers.serialize(tf.keras.regularizers.get(None)),
        "kernel_constraint": tf.keras.constraints.serialize(tf.keras.constraints.get(None)),
        "recurrent_constraint": tf.keras.constraints.serialize(tf.keras.constraints.get(None)),
        "bias_constraint": tf.keras.constraints.serialize(tf.keras.constraints.get(None)),
    }
    config = cell.get_config()
    assert config == expected_config

    restored_cell = RMCCell.from_config(config)
    restored_config = restored_cell.get_config()
    assert config == restored_config
