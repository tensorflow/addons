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
"""Tests for LayerNormSimpleRNN Cell."""

import numpy as np
import tensorflow.keras as keras

from tensorflow_addons.rnn import LayerNormSimpleRNNCell


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
