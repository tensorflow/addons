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
"""Tests for Novograd Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers import Novograd


@test_utils.run_all_in_graph_and_eager_modes
class NovogradTest(tf.test.TestCase):
    def run_dense_sample(self, iterations, expected, optimizer):
        var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
        var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

        grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
        grad_1 = tf.constant([0.3, 0.4], dtype=tf.dtypes.float32)

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        if tf.executing_eagerly():
            for _ in range(iterations):
                optimizer.apply_gradients(grads_and_vars)
        else:
            update = optimizer.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(iterations):
                self.evaluate(update)

        self.assertAllClose(var_0.read_value(), expected[0], atol=2e-4)
        self.assertAllClose(var_1.read_value(), expected[1], atol=2e-4)

    def run_sparse_sample(self, iterations, expected, optimizer):
        var_0 = tf.Variable([1.0, 2.0])
        var_1 = tf.Variable([3.0, 4.0])

        grad_0 = tf.IndexedSlices(
            tf.constant([0.1, 0.2]), tf.constant([0, 1]), tf.constant([2]))
        grad_1 = tf.IndexedSlices(
            tf.constant([0.3, 0.4]), tf.constant([0, 1]), tf.constant([2]))

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        if tf.executing_eagerly():
            for _ in range(iterations):
                optimizer.apply_gradients(grads_and_vars)
        else:
            update = optimizer.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(iterations):
                self.evaluate(update)

        self.assertAllClose(var_0.read_value(), expected[0], atol=2e-4)
        self.assertAllClose(var_1.read_value(), expected[1], atol=2e-4)

    def test_dense_sample(self):

        self.run_dense_sample(
            iterations=1,
            expected=[[0.8735088993, 1.7470177985], [2.8302943759, 3.7737258345]],
            optimizer=Novograd(lr=0.1),
        )

    def test_sparse_sample(self):
        self.run_sparse_sample(
            iterations=1,
            expected=[[0.8735088993, 1.7470177985], [2.8302943759, 3.7737258345]],
            optimizer=Novograd(lr=0.1),
        )

    def test_dense_sample_with_weight_decay(self):
        # Expected values are obtained from the official implementation
        self.run_dense_sample(
            iterations=2,
            expected=[[0.7382827095, 1.7470177985], [2.8302943759, 3.7737258345]],
            optimizer=Novograd(lr=0.1, weight_decay=0.01),
        )

    def test_sparse_sample_with_weight_decay(self):
        # Expected values are obtained from the official implementation
        # Dense results should be: [-0.2029,  0.7768], [1.7578, 2.7380]
        self.run_sparse_sample(
            iterations=1,
            expected=[[-0.2029, 2.0], [3.0, 2.7380]],
            optimizer=Novograd(lr=1e-3, weight_decay=0.01),
        )
        self.run_sparse_sample(
            iterations=2,
            expected=[[-0.2029, 2.0], [3.0, 2.7380]],
            optimizer=Novograd(lr=1e-3, weight_decay=0.01),
        )

    def test_dense_sample_with_grad_averaging(self):
        self.run_dense_sample(
            iterations=1,
            expected=[[0.8041, 1.8041], [2.8041, 3.8041]],
            optimizer=Novograd(lr=1e-3, grad_averaging=True))

    def test_sparse_sample_with_grad_averaging(self):
        self.run_sparse_sample(
            iterations=2,
            expected=[[0.4653, 2.0], [3.0, 3.4653]],
            optimizer=Novograd(lr=1e-3, grad_averaging=True))

    def test_get_config(self):
        opt = Novograd(lr=1e-4)
        config = opt.get_config()
        self.assertEqual(config['learning_rate'], 1e-4)
        self.assertEqual(config['weight_decay'], 0.0)
        self.assertEqual(config['grad_averaging'], False)


if __name__ == '__main__':
    tf.test.main()
