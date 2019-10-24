
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Stochastic Weight Averaging optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.optimizers import stochastic_weight_averaging
from tensorflow_addons.utils import test_utils

SWA = stochastic_weight_averaging.SWA

@test_utils.run_all_in_graph_and_eager_modes
class SWATest(tf.test.TestCase):

    def test_averaging(self):
        
        start_averaging = 0
        average_period = 1
        adam = tf.keras.optimizers.Adam(learning_rate=1)
        optimizer = SWA(adam, start_averaging, average_period)
              
        val_0 = [1., 1.]
        val_1 = [2., 2.]
        var_0 = tf.Variable(val_0)
        var_1 = tf.Variable(val_1)

        grad_val_0 = [0.1, 0.1]
        grad_val_1 = [0.1, 0.1]
        grad_0 = tf.constant(grad_val_0)
        grad_1 = tf.constant(grad_val_1)    
        grads_and_vars = zip([grad_0, grad_1], [var_0, var_1])
        if not tf.executing_eagerly():
            update = optimizer.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self.evaluate(update)
            self.evaluate(update)
        else:
            optimizer.apply_gradients(grads_and_vars)
            optimizer.apply_gradients(grads_and_vars)
        optimizer.assign_average_vars([var_0, var_1])
        
        # self.assertEqual(True, False, msg='{} | {}'.format(var_0, expected_var_0))
        self.assertAllClose(var_0.read_value(), [0.85, 0.85], msg='{}'.format(var_0.read_value()))
        self.assertAllClose(var_1.read_value(), [1.85, 1.85])

    def test_fit_simple_linear_model(self):
        np.random.seed(0x2019)
        num_examples = 100000
        x = np.random.standard_normal((num_examples, 3))
        w = np.random.standard_normal((3, 1))
        y = np.dot(x, w) + np.random.standard_normal((num_examples, 1)) * 1e-4

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
        # using num_examples - 1 since steps starts from 0.
        optimizer = SWA('adam',
                        start_averaging=num_examples // 32 - 1,
                        average_period=100)
        model.compile(optimizer, loss='mse')
        model.fit(x, y, epochs=3)
        optimizer.assign_average_vars(model.variables)
      
        x = np.random.standard_normal((100, 3))
        y = np.dot(x, w)

        predicted = model.predict(x)

        max_abs_diff = np.max(np.abs(predicted - y))
        self.assertLess(max_abs_diff, 1e-4)

    def test_optimizer_failure(self):
        with self.assertRaises(TypeError):
            _ = SWA(None, average_period=10)

    def test_optimizer_string(self):
        _ = SWA('adam', average_period=10)

    def test_get_config(self):
        self.skipTest('Wait #33614 to be fixed')
        opt = SWA('adam', average_period=10, start_averaging=0)
        opt = tf.keras.optimizers.deserialize(
            tf.keras.optimizers.serialize(opt))
        config = opt.get_config()
        self.assertEqual(config['average_period'], 10)
        self.assertEqual(config['start_averaging'], 0)


if __name__ == '__main__':
    tf.test.main()
