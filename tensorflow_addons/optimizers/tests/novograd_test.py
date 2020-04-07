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
"""Tests for NovoGrad Optimizer."""

import numpy as np
import tensorflow as tf

from tensorflow_addons.optimizers import NovoGrad
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class NovoGradTest(tf.test.TestCase):
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
            tf.constant([0.1, 0.2]), tf.constant([0, 1]), tf.constant([2])
        )
        grad_1 = tf.IndexedSlices(
            tf.constant([0.3, 0.4]), tf.constant([0, 1]), tf.constant([2])
        )

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
            expected=[[0.9552786425, 1.9105572849], [2.9400000012, 3.9200000016]],
            optimizer=NovoGrad(lr=0.1, epsilon=1e-8),
        )

    def test_sparse_sample(self):
        self.run_sparse_sample(
            iterations=1,
            expected=[[0.9552786425, 1.9105572849], [2.9400000012, 3.9200000016]],
            optimizer=NovoGrad(lr=0.1, epsilon=1e-8),
        )

    def test_dense_sample_with_weight_decay(self):
        self.run_dense_sample(
            iterations=1,
            expected=[[0.945278642, 1.8905572849], [2.9100000012, 3.8800000016]],
            optimizer=NovoGrad(lr=0.1, weight_decay=0.1, epsilon=1e-8),
        )

    def test_sparse_sample_with_weight_decay(self):
        self.run_sparse_sample(
            iterations=1,
            expected=[[0.945278642, 1.8905572849], [2.9100000012, 3.8800000016]],
            optimizer=NovoGrad(lr=0.1, weight_decay=0.1, epsilon=1e-8),
        )

    def test_dense_sample_with_grad_averaging(self):
        self.run_dense_sample(
            iterations=2,
            expected=[[0.9105572849, 1.8211145698], [2.8800000024, 3.8400000032]],
            optimizer=NovoGrad(lr=0.1, grad_averaging=True, epsilon=1e-8),
        )

    def test_sparse_sample_with_grad_averaging(self):
        self.run_sparse_sample(
            iterations=2,
            expected=[[0.9105572849, 1.8211145698], [2.8800000024, 3.8400000032]],
            optimizer=NovoGrad(lr=0.1, grad_averaging=True, epsilon=1e-8),
        )

    def test_fit_simple_linear_model(self):
        np.random.seed(0x2020)
        tf.random.set_seed(0x2020)

        x = np.random.standard_normal((100000, 3))
        w = np.random.standard_normal((3, 1))
        y = np.dot(x, w) + np.random.standard_normal((100000, 1)) * 1e-5

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
        model.compile(NovoGrad(), loss="mse")

        model.fit(x, y, epochs=2)

        x = np.random.standard_normal((100, 3))
        y = np.dot(x, w)
        predicted = model.predict(x)

        max_abs_diff = np.max(np.abs(predicted - y))
        self.assertLess(max_abs_diff, 1e-2)

    def test_get_config(self):
        opt = NovoGrad(lr=1e-4, weight_decay=0.0, grad_averaging=False)
        config = opt.get_config()
        self.assertEqual(config["learning_rate"], 1e-4)
        self.assertEqual(config["weight_decay"], 0.0)
        self.assertEqual(config["grad_averaging"], False)
