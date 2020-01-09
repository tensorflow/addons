# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for LayernormSimpleRNN layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.utils import test_utils
# from tensorflow_addons.rnn import LayernormSimpleRNN
from tensorflow_addons.rnn.layernorm_simplernn import LayernormSimpleRNN
# from layernorm_simplernn import LayernormSimpleRNN


@test_utils.run_all_in_graph_and_eager_modes
class LayernormSimpleRNNTest(tf.test.TestCase):
    def test_return_sequences_layernorm_rnn(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            LayernormSimpleRNN,
            kwargs={
                'units': units,
                'use_layernorm': True,
                'return_sequences': True
            },
            input_shape=(num_samples, timesteps, embedding_dim))

    def test_float64_layernorm_rnn(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            LayernormSimpleRNN,
            kwargs={
                'units': units,
                'use_layernorm': True,
                'return_sequences': True,
                'dtype': 'float64'
            },
            input_shape=(num_samples, timesteps, embedding_dim),
            input_dtype='float64')

    def test_dynamic_behavior_layernorm_rnn(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        layer = LayernormSimpleRNN(
            units, use_layernorm=True, input_shape=(None, embedding_dim))
        model = keras.models.Sequential()
        model.add(layer)
        model.compile('rmsprop', 'mse')
        x = np.random.random((num_samples, timesteps, embedding_dim))
        y = np.random.random((num_samples, units))
        model.train_on_batch(x, y)

    # DELETED TEST: test_implementation_mode_layernorm_rnn

    def test_dropout_layernorm_rnn(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        test_utils.layer_test(
            LayernormSimpleRNN,
            kwargs={
                'units': units,
                'use_layernorm': True,
                'dropout': 0.1,
                'recurrent_dropout': 0.1
            },
            input_shape=(num_samples, timesteps, embedding_dim))

    def test_constraints_layernorm_rnn(self):
        embedding_dim = 4
        layer_class = LayernormSimpleRNN
        k_constraint = keras.constraints.max_norm(0.01)
        r_constraint = keras.constraints.max_norm(0.01)
        b_constraint = keras.constraints.max_norm(0.01)
        g_constraint = keras.constraints.max_norm(0.01)
        layer = layer_class(
            5,
            use_layernorm=True,
            return_sequences=False,
            weights=None,
            input_shape=(None, embedding_dim),
            kernel_constraint=k_constraint,
            recurrent_constraint=r_constraint,
            bias_constraint=b_constraint,
            gamma_constraint=g_constraint)
        layer.build((None, None, embedding_dim))
        self.assertEqual(layer.cell.kernel.constraint, k_constraint)
        self.assertEqual(layer.cell.recurrent_kernel.constraint, r_constraint)
        self.assertEqual(layer.cell.bias.constraint, b_constraint)
        self.assertEqual(layer.cell.layernorm.gamma.constraint, g_constraint)

    def test_with_masking_layer_layernorm_rnn(self):
        layer_class = LayernormSimpleRNN
        inputs = np.random.random((2, 3, 4))
        targets = np.abs(np.random.random((2, 3, 5)))
        targets /= targets.sum(axis=-1, keepdims=True)
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(input_shape=(3, 4)))
        model.add(
            layer_class(
                units=5,
                use_layernorm=True,
                return_sequences=True,
                unroll=False))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

    def test_from_config_layernorm_rnn(self):
        layer_class = LayernormSimpleRNN
        for stateful in (False, True):
            l1 = layer_class(units=1, use_layernorm=True, stateful=stateful)
            l2 = layer_class.from_config(l1.get_config())
            assert l1.get_config() == l2.get_config()

    def test_regularizers_layernorm_rnn(self):
        embedding_dim = 4
        layer_class = LayernormSimpleRNN
        layer = layer_class(
            5,
            use_layernorm=True,
            return_sequences=False,
            weights=None,
            input_shape=(None, embedding_dim),
            kernel_regularizer=keras.regularizers.l1(0.01),
            recurrent_regularizer=keras.regularizers.l1(0.01),
            bias_regularizer='l2',
            gamma_regularizer='l2')
        # activity_regularizer='l1'  # DOESN'T DO ANYTHING
        layer.build((None, None, 2))
        self.assertEqual(len(layer.losses), 4)

        #x = keras.backend.variable(np.ones((2, 3, 2)))
        #layer(x)
        #if context.executing_eagerly():
        #    self.assertEqual(len(layer.losses), 4)
        #else:
        #    self.assertEqual(len(layer.get_losses_for(x)), 1)

    # STILL MISSING: test_statefulness_layernorm_rnn()


if __name__ == '__main__':
    tf.test.main()
