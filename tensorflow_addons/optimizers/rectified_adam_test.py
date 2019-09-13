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
"""Tests for Rectified Adam optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers import RectifiedAdam


@test_utils.run_all_in_graph_and_eager_modes
class RectifiedAdamTest(tf.test.TestCase):

    def run_dense_sample(self, iterations, expected, **opt_kwargs):
        var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
        var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

        grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
        grad_1 = tf.constant([0.03, 0.04], dtype=tf.dtypes.float32)

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        opt = RectifiedAdam(**opt_kwargs)

        if tf.executing_eagerly():
            for _ in range(iterations):
                opt.apply_gradients(grads_and_vars)
        else:
            update = opt.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(iterations):
                self.evaluate(update)

        self.assertAllClose(var_0.read_value(), expected[0], atol=1e-4)
        self.assertAllClose(var_1.read_value(), expected[1], atol=1e-4)

    def run_sparse_sample(self, iterations, expected, **opt_kwargs):
        var_0 = tf.Variable([1.0, 2.0])
        var_1 = tf.Variable([3.0, 4.0])

        grad_0 = tf.IndexedSlices(
            tf.constant([0.1]),
            tf.constant([0]),
            tf.constant([2])
        )
        grad_1 = tf.IndexedSlices(
            tf.constant([0.04]),
            tf.constant([1]),
            tf.constant([2])
        )

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        opt = RectifiedAdam(**opt_kwargs)

        if tf.executing_eagerly():
            for _ in range(iterations):
                opt.apply_gradients(grads_and_vars)
        else:
            update = opt.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(iterations):
                self.evaluate(update)

        self.assertAllClose(var_0.read_value(), expected[0], atol=1e-4)
        self.assertAllClose(var_1.read_value(), expected[1], atol=1e-4)

    def test_dense_sample(self):
        # Expected values are obtained from the official implementation
        self.run_dense_sample(
            iterations=1000,
            expected=[[0.5554, 1.5549], [2.5557, 3.5557]],
            lr=1e-3,
        )

    def test_sparse_sample(self):
        # Expected values are obtained from the official implementation
        # Dense results should be: [-0.1929,  0.8066], [1.8075, 2.8074]
        self.run_sparse_sample(
            iterations=2000,
            expected=[[-0.1929, 2.0], [3.0, 2.8074]],
            lr=1e-3,
        )

    def test_dense_sample_with_amsgrad(self):
        # Expected values are obtained from the official implementation
        # `amsgrad` has no effect because the gradient is fixed
        self.run_dense_sample(
            iterations=1000,
            expected=[[0.5554, 1.5549], [2.5557, 3.5557]],
            lr=1e-3,
            amsgrad=True,
        )

    def test_sparse_sample_with_amsgrad(self):
        # Expected values are obtained from the official implementation
        # `amsgrad` has no effect because the gradient is fixed
        self.run_sparse_sample(
            iterations=2000,
            expected=[[-0.1929, 2.0], [3.0, 2.8074]],
            lr=1e-3,
            amsgrad=True,
        )

    def test_dense_sample_with_weight_decay(self):
        # Expected values are obtained from the official implementation
        self.run_dense_sample(
            iterations=1000,
            expected=[[0.5472, 1.5368], [2.5276, 3.5176]],
            lr=1e-3,
            weight_decay=0.01,
        )

    def test_sparse_sample_with_weight_decay(self):
        # Expected values are obtained from the official implementation
        # Dense results should be: [-0.2029,  0.7768], [1.7578, 2.7380]
        self.run_sparse_sample(
            iterations=2000,
            expected=[[-0.2029, 2.0], [3.0, 2.7380]],
            lr=1e-3,
            weight_decay=0.01,
        )


if __name__ == '__main__':
    tf.test.main()
