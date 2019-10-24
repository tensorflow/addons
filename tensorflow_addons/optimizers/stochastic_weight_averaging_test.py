
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

    def run_sample(self, iterations, optimizer, seed=0x2019):
        np.random.seed(seed)

        val_0 = np.random.random((2,))
        val_1 = np.random.random((2,))

        var_0 = tf.Variable(val_0, dtype=tf.dtypes.float32)
        var_1 = tf.Variable(val_1, dtype=tf.dtypes.float32)

        grad_0 = tf.constant([0.1, 0.1])
        grad_1 = tf.constant([0.01, 0.01])

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        if tf.executing_eagerly():
            for _ in range(iterations):
                optimizer.apply_gradients(grads_and_vars)
        else:
            update = optimizer.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(iterations):
                self.evaluate(update)

        return [val_0, val_1], [grad_0, grad_1], [self.evaluate(var_0),
                                                  self.evaluate(var_1)]

    def test_averaging(self):
        for start_averaging in [0]:
            for average_period in [5, 10, 100]:
                optimizer = SWA('adam', start_averaging, average_period)
                initial_vals, grads, final_vals = self.run_sample(
                    start_averaging + average_period + 1, 
                    optimizer
                )
                first_vals = [
                    initial_vals[0] - (start_averaging + 1) * grads[0],
                    initial_vals[1] - (start_averaging + 1) * grads[1]
                ]
                average_vals = [tf.Variable((first_vals[0] + final_vals[0]) / 2.0),
                                tf.Variable((first_vals[1] + final_vals[1]) / 2.0)]
                optimizer.assign_average_vars(final_vals)
                msg = '{}'.format(final_vals)
                self.assertAllClose(average_vals, final_vals, msg=msg)

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
