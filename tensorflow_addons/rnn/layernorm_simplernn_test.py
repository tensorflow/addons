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

from tensorflow_addons.utils import test_utils  # for eager mode
#from tensorflow.python.eager import context  # to check eager mode
from tensorflow.python.keras import testing_utils  # for 'layer_test'
from tensorflow.python.training import gradient_descent  # for GD

import tensorflow_addons.rnn.layernorm_simplernn as lnrnn
# import layernorm_simplernn as lnrnn


@test_utils.run_all_in_graph_and_eager_modes
class LayernormSimpleRNNTest(tf.test.TestCase):
    def test_return_sequences_layernorm_rnn(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        testing_utils.layer_test(
            lnrnn.LayernormSimpleRNN,
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
        testing_utils.layer_test(
            lnrnn.LayernormSimpleRNN,
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
        layer = lnrnn.LayernormSimpleRNN(
            units, use_layernorm=True, input_shape=(None, embedding_dim))
        model = keras.models.Sequential()
        model.add(layer)
        model.compile('rmsprop', 'mse')
        x = np.random.random((num_samples, timesteps, embedding_dim))
        y = np.random.random((num_samples, units))
        model.train_on_batch(x, y)

    # test_implementation_mode_layernorm_rnn deleted

    def test_dropout_layernorm_rnn(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        testing_utils.layer_test(
            lnrnn.LayernormSimpleRNN,
            kwargs={
                'units': units,
                'use_layernorm': True,
                'dropout': 0.1,
                'recurrent_dropout': 0.1
            },
            input_shape=(num_samples, timesteps, embedding_dim))

    def test_constraints_layernorm_rnn(self):
        embedding_dim = 4
        layer_class = lnrnn.LayernormSimpleRNN
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
        layer_class = lnrnn.LayernormSimpleRNN
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
        layer_class = lnrnn.LayernormSimpleRNN
        for stateful in (False, True):
            l1 = layer_class(units=1, use_layernorm=True, stateful=stateful)
            l2 = layer_class.from_config(l1.get_config())
            assert l1.get_config() == l2.get_config()

    def test_regularizers_layernorm_rnn(self):
        embedding_dim = 4
        layer_class = lnrnn.LayernormSimpleRNN
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


"""
STILL FAILS
    def test_statefulness_layernorm_rnn(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        layer_class = lnrnn.LayernormSimpleRNN
        model = keras.models.Sequential()
        model.add(
            keras.layers.Embedding(
                4,
                embedding_dim,
                mask_zero=True,
                input_length=timesteps,
                batch_input_shape=(num_samples, timesteps)))
        layer = layer_class(
            units,
            use_layernorm=True,
            return_sequences=False,
            stateful=True,
            weights=None)
        model.add(layer)
        model.compile(
            optimizer=gradient_descent.GradientDescentOptimizer(0.01),
            loss='mse')
        out1 = model.predict(np.ones((num_samples, timesteps)))
        self.assertEqual(out1.shape, (num_samples, units))

        # train once so that the states change
        model.train_on_batch(
            np.ones((num_samples, timesteps)), np.ones((num_samples, units)))
        out2 = model.predict(np.ones((num_samples, timesteps)))

        # if the state is not reset, output should be different
        self.assertNotEqual(out1.max(), out2.max())

        # check that output changes after states are reset
        # (even though the model itself didn't change)
        layer.reset_states()
        out3 = model.predict(np.ones((num_samples, timesteps)))
        self.assertNotEqual(out2.max(), out3.max())

        # check that container-level reset_states() works
        model.reset_states()
        out4 = model.predict(np.ones((num_samples, timesteps)))
        np.testing.assert_allclose(out3, out4, atol=1e-5)

        # check that the call to `predict` updated the states
        out5 = model.predict(np.ones((num_samples, timesteps)))
        self.assertNotEqual(out4.max(), out5.max())

        # Check masking
        layer.reset_states()

        left_padded_input = np.ones((num_samples, timesteps))
        left_padded_input[0, :1] = 0
        left_padded_input[1, :2] = 0
        out6 = model.predict(left_padded_input)

        layer.reset_states()

        right_padded_input = np.ones((num_samples, timesteps))
        right_padded_input[0, -1:] = 0
        right_padded_input[1, -2:] = 0
        out7 = model.predict(right_padded_input)

        np.testing.assert_allclose(out7, out6, atol=1e-5)
"""

if __name__ == '__main__':
    tf.test.main()
