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
"""Tests for Proximal Adagrad optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers import ProximalAdagrad


@test_utils.run_all_in_graph_and_eager_modes
class ProximalAdagradTest(tf.test.TestCase):
    def test_without_regularization(self):
        with tf.device("cpu"):
            var0 = tf.Variable([0.0, 0.0])
            var1 = tf.Variable([0.0, 0.0])
            grads0 = tf.constant([0.1, 0.2])
            grads1 = tf.constant([0.01, 0.02])
            opt = ProximalAdagrad(
                3.0,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.0,
                l2_regularization_strength=0.0)

            if not tf.executing_eagerly():
                update = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            v0_val, v1_val = self.evaluate([var0, var1])
            self.assertAllClose([0.0, 0.0], v0_val)
            self.assertAllClose([0.0, 0.0], v1_val)

            # Run 3 steps Proximal Adagrad.
            for _ in range(3):
                if not tf.executing_eagerly():
                    self.evaluate(update)
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            v0_val, v1_val = self.evaluate([var0, var1])
            self.assertAllClose([-2.60260963, -4.29698515], v0_val)
            self.assertAllClose([-0.28432083, -0.56694895], v1_val)

    def test_with_l1_regularization(self):
        with tf.device("cpu"):
            var0 = tf.Variable([1.0, 2.0])
            var1 = tf.Variable([4.0, 3.0])
            grads0 = tf.constant([0.1, 0.2])
            grads1 = tf.constant([0.01, 0.02])
            opt = ProximalAdagrad(
                3.0,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.001,
                l2_regularization_strength=0.0)

            if not tf.executing_eagerly():
                update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            v0_val, v1_val = self.evaluate([var0, var1])
            self.assertAllClose([1.0, 2.0], v0_val)
            self.assertAllClose([4.0, 3.0], v1_val)

            # Run 10 steps Proximal Adagrad.
            for _ in range(10):
                if not tf.executing_eagerly():
                    self.evaluate(update)
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            v0_val, v1_val = self.evaluate([var0, var1])
            self.assertAllClose([-6.663634, -9.190331], v0_val)
            self.assertAllClose([2.959304, 1.029232], v1_val)

    def test_with_l1_l2_regularization(self):
        with tf.device("cpu"):
            var0 = tf.Variable([1.0, 2.0])
            var1 = tf.Variable([4.0, 3.0])
            grads0 = tf.constant([0.1, 0.2])
            grads1 = tf.constant([0.01, 0.02])
            opt = ProximalAdagrad(
                3.0,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.001,
                l2_regularization_strength=2.0)

            if not tf.executing_eagerly():
                update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            v0_val, v1_val = self.evaluate([var0, var1])
            self.assertAllClose([1.0, 2.0], v0_val)
            self.assertAllClose([4.0, 3.0], v1_val)

            # Run 10 steps Proximal Adagrad.
            for _ in range(10):
                if not tf.executing_eagerly():
                    self.evaluate(update)
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            v0_val, v1_val = self.evaluate([var0, var1])
            self.assertAllClose([-0.0495, -0.0995], v0_val)
            self.assertAllClose([-0.0045, -0.0095], v1_val)


if __name__ == '__main__':
    tf.test.main()
