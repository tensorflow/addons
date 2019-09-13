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

    def test_dense_sample(self):
        var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
        var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

        grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
        grad_1 = tf.constant([0.03, 0.04], dtype=tf.dtypes.float32)

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        opt = RectifiedAdam(lr=1e-3)

        if tf.executing_eagerly():
            for _ in range(1000):
                opt.apply_gradients(grads_and_vars)
        else:
            update = opt.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(1000):
                self.evaluate(update)

        # Expected values are obtained from the official implementation
        self.assertAllClose(var_0.read_value(), [0.5554, 1.5549], atol=1e-4)
        self.assertAllClose(var_1.read_value(), [2.5557, 3.5557], atol=1e-4)

    def test_sparse_sample(self):
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

        opt = RectifiedAdam(lr=1e-3)

        if tf.executing_eagerly():
            for _ in range(5000):
                opt.apply_gradients(grads_and_vars)
        else:
            update = opt.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(5000):
                self.evaluate(update)

        # Expected values are obtained from the official implementation
        # Dense results should be: [-2.9875, -1.9880], [-0.9871,  0.0128]
        self.assertAllClose(var_0.read_value(), [-2.9875, 2.0], atol=1e-4)
        self.assertAllClose(var_1.read_value(), [3.0, 0.0128], atol=1e-4)


if __name__ == '__main__':
    tf.test.main()
