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
"""Tests for MovingAverage optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_addons.optimizers import MovingAverage
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class MovingAverageTest(tf.test.TestCase):
    def test_run(self):
        for sequential_update in [True, False]:
            var0 = tf.Variable([1.0, 2.0])
            var1 = tf.Variable([3.0, 4.0])

            grads0 = tf.constant([0.1, 0.1])
            grads1 = tf.constant([0.01, 0.01])

            grads_and_vars = list(zip([grads0, grads1], [var0, var1]))

            opt = MovingAverage(
                tf.keras.optimizers.SGD(lr=2.0),
                average_decay=0.5,
                sequential_update=sequential_update)

            if not tf.executing_eagerly():
                update = opt.apply_gradients(grads_and_vars)
                self.evaluate(tf.compat.v1.global_variables_initializer())
                self.evaluate(update)
                self.evaluate(update)
            else:
                opt.apply_gradients(grads_and_vars)
                opt.apply_gradients(grads_and_vars)

            self.assertAllClose(var0.read_value(), [0.6, 1.6])
            self.assertAllClose(var1.read_value(), [2.96, 3.96])

            ema_var0 = opt._ema.average(var0)  # pylint: disable=protected-access
            ema_var1 = opt._ema.average(var1)  # pylint: disable=protected-access

            if sequential_update:
                self.assertAllClose(ema_var0.read_value(), [0.75, 1.75])
                self.assertAllClose(ema_var1.read_value(), [2.975, 3.975])

            assign = opt.assign_average_vars([var0, var1])
            self.evaluate(assign)

            if sequential_update:
                self.assertAllClose(var0.read_value(), [0.75, 1.75])
                self.assertAllClose(var1.read_value(), [2.975, 3.975])

            perturb = tf.group([
                var0.assign_add([1.0, 1.0]),
                var1.assign_add([2.0, 2.0]),
                ema_var0.assign_add([3.0, 3.0]),
                ema_var1.assign_add([4.0, 4.0])
            ])
            self.evaluate(perturb)

            if sequential_update:
                self.assertAllClose(var0.read_value(), [1.75, 2.75])
                self.assertAllClose(var1.read_value(), [4.975, 5.975])
                self.assertAllClose(ema_var0.read_value(), [3.75, 4.75])
                self.assertAllClose(ema_var1.read_value(), [6.975, 7.975])

    def test_opt_failure(self):
        base_opt = None
        for sequential_update in [True, False]:
            with self.assertRaises(TypeError):
                MovingAverage(base_opt, 0.5, sequential_update)

    def test_model_weights_update(self):
        grad = tf.Variable([[0.1]])
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.keras.initializers.Constant([[1.0]]),
                use_bias=False)
        ])
        model.build(input_shape=[1, 1])

        opt = MovingAverage(tf.keras.optimizers.SGD(lr=2.0), 0.5)
        update = opt.apply_gradients(list(zip([grad], model.variables)))

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(update)
        self.assertAllClose(model.variables[0].read_value(), [[0.8]])

        mean_update = opt.assign_average_vars(model.variables)
        self.evaluate(mean_update)
        self.assertAllClose(model.variables[0].read_value(), [[0.9]])

    def test_optimizer_string(self):
        _ = MovingAverage('adam')

    def test_config(self):
        sgd_opt = tf.keras.optimizers.SGD(
            lr=2.0, nesterov=True, momentum=0.3, decay=0.1)
        opt = MovingAverage(
            sgd_opt,
            average_decay=0.5,
            num_updates=None,
            sequential_update=False)
        config = opt.get_config()

        self.assertEqual(config['average_decay'], 0.5)
        self.assertEqual(config['num_updates'], None)
        self.assertEqual(config['sequential_update'], False)

        new_opt = MovingAverage.from_config(config)
        old_sgd_config = opt._optimizer.get_config()
        new_sgd_config = new_opt._optimizer.get_config()

        for k1, k2 in zip(old_sgd_config, new_sgd_config):
            self.assertEqual(old_sgd_config[k1], new_sgd_config[k2])


if __name__ == '__main__':
    tf.test.main()
