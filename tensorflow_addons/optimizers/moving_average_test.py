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

import six

import tensorflow as tf

import moving_average
from tensorflow_addons.utils import test_utils


class MovingAverageTest(tf.test.TestCase):
    @test_utils.run_deprecated_v1
    def test_run(self):
        for seq_update in [True, False]:
            orig_var0 = [1.0, 2.0]
            orig_var1 = [3.0, 4.0]

            var0 = tf.Variable(orig_var0)
            var1 = tf.Variable(orig_var1)

            grads0 = tf.constant([0.1, 0.1])
            grads1 = tf.constant([0.01, 0.01])

            opt = moving_average.MovingAverage(
                tf.keras.optimizers.SGD(lr=2.0),
                average_decay=0.5,
                seq_update=seq_update)

            update = opt.apply_gradients(
                list(six.moves.zip([grads0, grads1], [var0, var1])))

            ema_var0 = opt._ema.average(var0)  # pylint: disable=protected-access
            ema_var1 = opt._ema.average(var1)  # pylint: disable=protected-access

            self.evaluate(tf.compat.v1.global_variables_initializer())
            self.evaluate(update)

            self.assertAllClose(var0.read_value(), [0.8, 1.8])
            self.assertAllClose(var1.read_value(), [2.98, 3.98])

            if seq_update:
                self.assertAllClose(ema_var0.read_value(), [0.9, 1.9])
                self.assertAllClose(ema_var1.read_value(), [2.99, 3.99])

            assign = opt.assign_average_vars([var0, var1])
            self.evaluate(assign)

            if seq_update:
                self.assertAllClose(self.evaluate(var0), [0.9, 1.9])
                self.assertAllClose(self.evaluate(var1), [2.99, 3.99])

            perturb = tf.group([
                var0.assign_add([1.0, 1.0]),
                var1.assign_add([2.0, 2.0]),
                ema_var0.assign_add([3.0, 3.0]),
                ema_var1.assign_add([4.0, 4.0])
            ])
            self.evaluate(perturb)

            if seq_update:
                self.assertAllClose(self.evaluate(var0), [1.9, 2.9])
                self.assertAllClose(self.evaluate(var1), [4.99, 5.99])
                self.assertAllClose(self.evaluate(ema_var0), [3.9, 4.9])
                self.assertAllClose(self.evaluate(ema_var1), [6.99, 7.99])

    @test_utils.run_in_graph_and_eager_modes
    def test_opt_failure(self):
        base_opt = None
        for seq_update in [True, False]:
            with self.assertRaises(TypeError):
                moving_average.MovingAverage(base_opt, 0.5, seq_update)

    @test_utils.run_deprecated_v1
    def test_model_weights_update(self):
        grad = tf.Variable([[0.1]])
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.keras.initializers.Constant([[1.0]]),
                use_bias=False)
        ])

        model.build(input_shape=[1, 1])

        opt = moving_average.MovingAverage(
            tf.keras.optimizers.SGD(lr=2.0), 0.5)

        update = opt.apply_gradients(
            list(six.moves.zip([grad], model.variables)))

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(update)
        self.assertAllClose(model.variables[0].read_value(), [[0.8]])

        mean_update = opt.assign_average_vars(model.variables)
        self.evaluate(mean_update)
        self.assertAllClose(model.variables[0].read_value(), [[0.9]])


if __name__ == '__main__':
    tf.test.main()
