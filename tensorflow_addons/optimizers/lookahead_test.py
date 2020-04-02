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
"""Tests for Lookahead optimizer."""

import numpy as np
import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers import Lookahead


@test_utils.run_all_in_graph_and_eager_modes
class LookaheadTest(tf.test.TestCase):
    def run_dense_sample(self, iterations, optimizer, seed=0x2019):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        val_0 = np.random.random((2,))
        val_1 = np.random.random((2,))

        var_0 = tf.Variable(val_0, dtype=tf.dtypes.float32)
        var_1 = tf.Variable(val_1, dtype=tf.dtypes.float32)

        grad_0 = tf.constant(np.random.standard_normal((2,)), dtype=tf.dtypes.float32)
        grad_1 = tf.constant(np.random.standard_normal((2,)), dtype=tf.dtypes.float32)

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        if tf.executing_eagerly():
            for _ in range(iterations):
                optimizer.apply_gradients(grads_and_vars)
        else:
            update = optimizer.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(iterations):
                self.evaluate(update)

        return [val_0, val_1], [self.evaluate(var_0), self.evaluate(var_1)]

    def run_sparse_sample(self, iterations, optimizer, seed=0x2019):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        val_0 = np.random.random((2,))
        val_1 = np.random.random((2,))

        var_0 = tf.Variable(val_0, dtype=tf.dtypes.float32)
        var_1 = tf.Variable(val_1, dtype=tf.dtypes.float32)

        grad_0 = tf.IndexedSlices(
            tf.constant([np.random.standard_normal()]),
            tf.constant([0]),
            tf.constant([2]),
        )
        grad_1 = tf.IndexedSlices(
            tf.constant([np.random.standard_normal()]),
            tf.constant([1]),
            tf.constant([2]),
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

        return [val_0, val_1], [self.evaluate(var_0), self.evaluate(var_1)]

    def test_dense_exact_ratio(self):
        for k in [5, 10, 100]:
            for alpha in [0.3, 0.7]:
                optimizer = tf.keras.optimizers.get("adam")
                vals, quick_vars = self.run_dense_sample(k, optimizer)
                optimizer = Lookahead("adam", sync_period=k, slow_step_size=alpha)
                _, slow_vars = self.run_dense_sample(k, optimizer)
                for val, quick, slow in zip(vals, quick_vars, slow_vars):
                    expected = val + (quick - val) * alpha
                    self.assertAllClose(expected, slow)

    def test_sparse_exact_ratio(self):
        for k in [5, 10, 100]:
            for alpha in [0.3, 0.7]:
                optimizer = tf.keras.optimizers.get("adam")
                vals, quick_vars = self.run_sparse_sample(k, optimizer)
                optimizer = Lookahead("adam", sync_period=k, slow_step_size=alpha)
                _, slow_vars = self.run_sparse_sample(k, optimizer)
                for val, quick, slow in zip(vals, quick_vars, slow_vars):
                    expected = val + (quick - val) * alpha
                    self.assertAllClose(expected, slow)

    def test_fit_simple_linear_model(self):
        np.random.seed(0x2019)
        tf.random.set_seed(0x2019)

        x = np.random.standard_normal((10000, 3))
        w = np.random.standard_normal((3, 1))
        y = np.dot(x, w) + np.random.standard_normal((10000, 1)) * 1e-4

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
        model.compile(Lookahead("sgd"), loss="mse")

        model.fit(x, y, epochs=3)

        x = np.random.standard_normal((100, 3))
        y = np.dot(x, w)
        predicted = model.predict(x)

        max_abs_diff = np.max(np.abs(predicted - y))
        self.assertLess(max_abs_diff, 1e-3)

    def test_model_dynamic_lr(self):
        grad = tf.Variable([[0.1]])
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    1,
                    kernel_initializer=tf.keras.initializers.Constant([[1.0]]),
                    use_bias=False,
                )
            ]
        )
        model.build(input_shape=[1, 1])

        opt = Lookahead("adam", sync_period=10, slow_step_size=0.4)
        update = opt.apply_gradients(list(zip([grad], model.variables)))

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(update)
        self.assertAllClose(opt.lr.read_value(), 1e-3)

        opt.lr = 1e-4
        self.assertAllClose(opt.lr.read_value(), 1e-4)

    def test_get_config(self):
        opt = Lookahead("adam", sync_period=10, slow_step_size=0.4)
        opt = tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(opt))
        config = opt.get_config()
        self.assertEqual(config["sync_period"], 10)
        self.assertEqual(config["slow_step_size"], 0.4)
